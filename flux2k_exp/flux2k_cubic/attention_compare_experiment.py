"""Standalone attention comparison experiment for Flux2K cubic.

This module keeps all three-scenario comparison logic in one place:
1. full baseline
2. selective without KV
3. selective with stale frozen KV from the last anchor step
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

Path("/tmp/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib
import numpy as np
import torch

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb

from cubic_core import AdaptivePartitioner, CurvatureMonitor, Phase
from flux2_klein.pipeline import Flux2KleinPipeline, compute_empirical_mu, retrieve_timesteps
from flux2_klein.transformer import _get_qkv_projections, Flux2Attention
from .optimizer import Flux2KAdaptiveOptimizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _call_cache_update(
    engine,
    cache_runtime: dict,
    layer_key: tuple[str, int],
    key_active: torch.Tensor,
    value_active: torch.Tensor,
    active_idx: torch.Tensor,
    total_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return engine._update_and_get_image_kv_cache(
        cache_key=cache_runtime["cache_key"],
        layer_key=layer_key,
        key_active=key_active,
        value_active=value_active,
        active_idx=active_idx,
        total_tokens=total_tokens,
    )


class CompareAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, engine, layer_kind: str, layer_idx: int):
        self.engine = engine
        self.layer_kind = layer_kind
        self.layer_idx = layer_idx

    def __call__(
        self,
        attn: Flux2Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ):
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            self.engine.capture_layer_output(
                (self.layer_kind, self.layer_idx),
                torch.cat([encoder_hidden_states, hidden_states], dim=1),
            )
            return hidden_states, encoder_hidden_states

        self.engine.capture_layer_output((self.layer_kind, self.layer_idx), hidden_states)
        return hidden_states


class CompareParallelSelfAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, engine, layer_kind: str, layer_idx: int):
        self.engine = engine
        self.layer_kind = layer_kind
        self.layer_idx = layer_idx

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        projected = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            projected, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        attn_output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        attn_output = attn_output.flatten(2, 3).to(query.dtype)

        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)

        self.engine.capture_layer_output((self.layer_kind, self.layer_idx), hidden_states)
        return hidden_states


class CompareStaleKVAttnProcessor(CompareAttnProcessor):
    def __call__(
        self,
        attn: Flux2Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ):
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        cache_runtime = self.engine.runtime
        text_len = int(encoder_hidden_states.shape[1]) if encoder_hidden_states is not None else 0
        img_active_idx = cache_runtime["active_idx"]
        img_total_tokens = int(cache_runtime["total_tokens"])

        if text_len > 0:
            text_idx = torch.arange(text_len, device=img_active_idx.device, dtype=img_active_idx.dtype)
            seq_active_idx = torch.cat([text_idx, img_active_idx + text_len], dim=0)
        else:
            seq_active_idx = img_active_idx

        key, value = _call_cache_update(
            engine=self.engine,
            cache_runtime=cache_runtime,
            layer_key=(self.layer_kind, self.layer_idx),
            key_active=key,
            value_active=value,
            active_idx=seq_active_idx,
            total_tokens=text_len + img_total_tokens,
        )
        return self._finish_with_cached(
            attn=attn,
            query=query,
            key=key,
            value=value,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )

    def _finish_with_cached(
        self,
        attn: Flux2Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ):
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            self.engine.capture_layer_output(
                (self.layer_kind, self.layer_idx),
                torch.cat([encoder_hidden_states, hidden_states], dim=1),
            )
            return hidden_states, encoder_hidden_states

        self.engine.capture_layer_output((self.layer_kind, self.layer_idx), hidden_states)
        return hidden_states


class CompareStaleKVParallelSelfAttnProcessor(CompareParallelSelfAttnProcessor):
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        projected = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            projected, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        cache_runtime = self.engine.runtime
        encoder_tokens = int(cache_runtime.get("encoder_tokens", 0))
        img_active_idx = cache_runtime["active_idx"]
        img_total_tokens = int(cache_runtime["total_tokens"])

        if encoder_tokens > 0:
            text_idx = torch.arange(encoder_tokens, device=img_active_idx.device, dtype=img_active_idx.dtype)
            seq_active_idx = torch.cat([text_idx, img_active_idx + encoder_tokens], dim=0)
        else:
            seq_active_idx = img_active_idx

        key, value = _call_cache_update(
            engine=self.engine,
            cache_runtime=cache_runtime,
            layer_key=(self.layer_kind, self.layer_idx),
            key_active=key,
            value_active=value,
            active_idx=seq_active_idx,
            total_tokens=encoder_tokens + img_total_tokens,
        )

        attn_output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        attn_output = attn_output.flatten(2, 3).to(query.dtype)

        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)
        self.engine.capture_layer_output((self.layer_kind, self.layer_idx), hidden_states)
        return hidden_states


class Flux2KAttentionCompareEngine:
    def __init__(self, use_kv_cache: bool):
        self.use_kv_cache = bool(use_kv_cache)
        self.output_cache: dict[str, torch.Tensor] = {}
        self.k_cache: dict[tuple[str, str, int], torch.Tensor] = {}
        self.v_cache: dict[tuple[str, str, int], torch.Tensor] = {}
        self.anchor_k_cache: dict[tuple[str, str, int], torch.Tensor] = {}
        self.anchor_v_cache: dict[tuple[str, str, int], torch.Tensor] = {}
        self._capture_session: dict | None = None
        self._last_capture_session: dict | None = None
        self._runtime: dict | None = None
        self._expected_layer_keys: list[tuple[str, int]] = []
        self._expected_layer_shapes: dict[tuple[str, int], tuple[int, int]] = {}

    @property
    def runtime(self) -> dict:
        if self._runtime is None:
            raise RuntimeError("runtime is not available")
        return self._runtime

    def reset(self) -> None:
        self.output_cache.clear()
        self.k_cache.clear()
        self.v_cache.clear()
        self.anchor_k_cache.clear()
        self.anchor_v_cache.clear()
        self._capture_session = None
        self._last_capture_session = None
        self._runtime = None

    def snapshot_anchor_cache(self, cache_key: str | None = None) -> None:
        self.anchor_k_cache.clear()
        self.anchor_v_cache.clear()
        for key, value in self.k_cache.items():
            if cache_key is None or key[0] == cache_key:
                self.anchor_k_cache[key] = value.detach().clone()
        for key, value in self.v_cache.items():
            if cache_key is None or key[0] == cache_key:
                self.anchor_v_cache[key] = value.detach().clone()

    def pop_last_capture(self) -> dict | None:
        capture = self._last_capture_session
        self._last_capture_session = None
        return capture

    def forward(
        self,
        transformer,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance,
        pooled_projections,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor | None,
        joint_attention_kwargs,
        step_plan=None,
        cache_key: str = "default",
        step_idx: int | None = None,
        step_phase: str | None = None,
        trace_label: str | None = None,
        capture_label: str | None = None,
        frozen_cache_mode: str = "rolling",
    ) -> torch.Tensor:
        del pooled_projections

        use_selective = step_plan is not None and (not step_plan.is_full_compute)
        seq_len = int(hidden_states.shape[1])
        text_tokens = int(encoder_hidden_states.shape[1])
        branch_label = trace_label or cache_key

        if not use_selective:
            return self._full_forward_and_refresh_cache(
                transformer=transformer,
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                cache_key=cache_key,
                step_idx=step_idx,
                step_phase=step_phase,
                trace_label=branch_label,
                capture_label=capture_label,
                reason="no_selective_plan",
                frozen_cache_mode=frozen_cache_mode,
            )

        active_idx = torch.unique(step_plan.active_token_indices.long().to(hidden_states.device), sorted=True)
        active_count = int(active_idx.numel())
        cached = self.output_cache.get(cache_key)

        if active_count == 0:
            if cached is None or cached.shape != hidden_states.shape:
                return self._full_forward_and_refresh_cache(
                    transformer=transformer,
                    hidden_states=hidden_states,
                    timestep=timestep,
                    guidance=guidance,
                    encoder_hidden_states=encoder_hidden_states,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    cache_key=cache_key,
                    step_idx=step_idx,
                    step_phase=step_phase,
                    trace_label=branch_label,
                    capture_label=capture_label,
                    reason="empty_active_set_without_output_cache",
                    frozen_cache_mode=frozen_cache_mode,
                )
            _trace(step_idx, branch_label, step_phase or "selective", f"forward mode={'selective+kv' if self.use_kv_cache else 'selective'} img_active=0/{seq_len} txt={text_tokens} cache=reuse")
            return cached.clone()

        if active_count >= seq_len or cached is None or cached.shape != hidden_states.shape:
            reason = "all_tokens_active" if active_count >= seq_len else (
                "output_cache_miss" if cached is None else "output_cache_shape_mismatch"
            )
            return self._full_forward_and_refresh_cache(
                transformer=transformer,
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                cache_key=cache_key,
                step_idx=step_idx,
                step_phase=step_phase,
                trace_label=branch_label,
                capture_label=capture_label,
                reason=reason,
                frozen_cache_mode=frozen_cache_mode,
            )

        if self.use_kv_cache and not self._has_kv_cache_for_all_layers(
            transformer=transformer,
            cache_key=cache_key,
            batch_size=int(hidden_states.shape[0]),
            image_tokens=seq_len,
            encoder_tokens=text_tokens,
            device=hidden_states.device,
        ):
            return self._full_forward_and_refresh_cache(
                transformer=transformer,
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                cache_key=cache_key,
                step_idx=step_idx,
                step_phase=step_phase,
                trace_label=branch_label,
                capture_label=capture_label,
                reason="kv_cache_incomplete",
                frozen_cache_mode=frozen_cache_mode,
            )

        active_hidden = hidden_states.index_select(dim=1, index=active_idx)
        selected_img_ids = _slice_img_ids(img_ids, active_idx)
        _trace(
            step_idx,
            branch_label,
            step_phase or "selective",
            f"forward mode={'selective+kv' if self.use_kv_cache else 'selective'} img_active={active_count}/{seq_len} txt={text_tokens} cache=hit",
        )

        if self.use_kv_cache:
            active_out = self._run_with_stale_kv(
                transformer=transformer,
                hidden_states=active_hidden,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=selected_img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                cache_key=cache_key,
                active_idx=active_idx,
                total_tokens=seq_len,
                encoder_tokens=text_tokens,
                step_idx=step_idx,
                step_phase=step_phase,
                trace_label=branch_label,
                capture_label=capture_label,
                frozen_cache_mode=frozen_cache_mode,
            )
        else:
            active_out = self._execute_transformer(
                transformer=transformer,
                hidden_states=active_hidden,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=selected_img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                capture_label=capture_label,
                runtime=None,
            )

        output = cached.clone()
        output[:, active_idx, :] = active_out
        self.output_cache[cache_key] = output.detach()
        return output

    def _full_forward_and_refresh_cache(
        self,
        transformer,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor | None,
        joint_attention_kwargs,
        cache_key: str,
        step_idx: int | None,
        step_phase: str | None,
        trace_label: str,
        capture_label: str | None,
        reason: str,
        frozen_cache_mode: str,
    ) -> torch.Tensor:
        seq_len = int(hidden_states.shape[1])
        text_tokens = int(encoder_hidden_states.shape[1])
        mode = "full+kv" if self.use_kv_cache else "full"
        _trace(step_idx, trace_label, step_phase or "full", f"forward mode={mode} img_active={seq_len}/{seq_len} txt={text_tokens} reason={reason}")

        if self.use_kv_cache:
            active_idx = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long)
            output = self._run_with_stale_kv(
                transformer=transformer,
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                cache_key=cache_key,
                active_idx=active_idx,
                total_tokens=seq_len,
                encoder_tokens=text_tokens,
                step_idx=step_idx,
                step_phase=step_phase,
                trace_label=trace_label,
                capture_label=capture_label,
                frozen_cache_mode=frozen_cache_mode,
            )
        else:
            output = self._execute_transformer(
                transformer=transformer,
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                capture_label=capture_label,
                runtime=None,
            )

        self.output_cache[cache_key] = output.detach()
        return output

    def _run_with_stale_kv(
        self,
        transformer,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor | None,
        joint_attention_kwargs,
        cache_key: str,
        active_idx: torch.Tensor,
        total_tokens: int,
        encoder_tokens: int,
        step_idx: int | None,
        step_phase: str | None,
        trace_label: str,
        capture_label: str | None,
        frozen_cache_mode: str,
    ) -> torch.Tensor:
        runtime = {
            "cache_key": cache_key,
            "active_idx": active_idx,
            "total_tokens": int(total_tokens),
            "encoder_tokens": int(encoder_tokens),
            "step_idx": step_idx,
            "step_phase": step_phase,
            "trace_label": trace_label,
            "forward_mode": "selective+kv" if int(active_idx.numel()) < int(total_tokens) else "full+kv",
            "active_image_tokens": int(active_idx.numel()),
            "frozen_cache_mode": frozen_cache_mode,
        }
        return self._execute_transformer(
            transformer=transformer,
            hidden_states=hidden_states,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            capture_label=capture_label,
            runtime=runtime,
        )

    def _execute_transformer(
        self,
        transformer,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor | None,
        joint_attention_kwargs,
        capture_label: str | None,
        runtime: dict | None,
    ) -> torch.Tensor:
        prev_runtime = self._runtime
        prev_capture = self._capture_session
        self._ensure_processors(transformer)
        if runtime is not None:
            self._runtime = runtime
        if capture_label is not None:
            self._capture_session = {"label": capture_label, "layers": {}}

        try:
            output = transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=False,
            )[0]
            if capture_label is not None and self._capture_session is not None:
                self._last_capture_session = {
                    "label": capture_label,
                    "layers": self._capture_session["layers"],
                }
            return output
        finally:
            self._runtime = prev_runtime
            self._capture_session = prev_capture

    def _ensure_processors(self, transformer) -> None:
        self._expected_layer_keys = []
        self._expected_layer_shapes = {}
        attn_cls = CompareStaleKVAttnProcessor if self.use_kv_cache else CompareAttnProcessor
        single_cls = CompareStaleKVParallelSelfAttnProcessor if self.use_kv_cache else CompareParallelSelfAttnProcessor

        for idx, block in enumerate(transformer.transformer_blocks):
            block.attn.set_processor(attn_cls(self, layer_kind="joint", layer_idx=idx))
            self._expected_layer_keys.append(("joint", idx))
            self._expected_layer_shapes[("joint", idx)] = (int(block.attn.heads), int(block.attn.head_dim))

        for idx, block in enumerate(transformer.single_transformer_blocks):
            block.attn.set_processor(single_cls(self, layer_kind="single", layer_idx=idx))
            self._expected_layer_keys.append(("single", idx))
            self._expected_layer_shapes[("single", idx)] = (int(block.attn.heads), int(block.attn.head_dim))

    def capture_layer_output(self, layer_key: tuple[str, int], attn_output: torch.Tensor) -> None:
        if self._capture_session is None:
            return
        layer_record = self._capture_session["layers"].setdefault(layer_key, {})
        layer_record["attn_output"] = attn_output.detach().clone()

    def _update_and_get_image_kv_cache(
        self,
        cache_key: str,
        layer_key: tuple[str, int],
        key_active: torch.Tensor,
        value_active: torch.Tensor,
        active_idx: torch.Tensor,
        total_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_id = (cache_key, layer_key[0], layer_key[1])
        k_cache = self.k_cache.get(cache_id)
        v_cache = self.v_cache.get(cache_id)

        cache_shape_ok = (
            k_cache is not None
            and v_cache is not None
            and k_cache.shape == v_cache.shape
            and k_cache.dtype == v_cache.dtype
            and k_cache.device == v_cache.device
            and k_cache.shape[0] == key_active.shape[0]
            and k_cache.shape[1] == int(total_tokens)
            and k_cache.shape[2] == key_active.shape[2]
            and k_cache.shape[3] == key_active.shape[3]
        )

        frozen_cache_mode = str(self.runtime.get("frozen_cache_mode", "rolling"))
        if frozen_cache_mode == "anchor_base" and int(key_active.shape[1]) != int(total_tokens):
            anchor_k = self.anchor_k_cache.get(cache_id)
            anchor_v = self.anchor_v_cache.get(cache_id)
            anchor_shape_ok = (
                anchor_k is not None
                and anchor_v is not None
                and anchor_k.shape == anchor_v.shape
                and anchor_k.dtype == anchor_v.dtype
                and anchor_k.device == anchor_v.device
                and anchor_k.shape[0] == key_active.shape[0]
                and anchor_k.shape[1] == int(total_tokens)
                and anchor_k.shape[2] == key_active.shape[2]
                and anchor_k.shape[3] == key_active.shape[3]
            )
            if anchor_shape_ok:
                k_cache = anchor_k.detach().clone()
                v_cache = anchor_v.detach().clone()
                k_cache[:, active_idx, :, :] = key_active
                v_cache[:, active_idx, :, :] = value_active
                self.k_cache[cache_id] = k_cache.detach()
                self.v_cache[cache_id] = v_cache.detach()
                self._trace_layer(layer_key, int(key_active.shape[1]), int(total_tokens), "anchor_update")
                return k_cache, v_cache

        if not cache_shape_ok:
            k_cache = key_active.detach()
            v_cache = value_active.detach()
            cache_event = "refresh"
        else:
            k_cache[:, active_idx, :, :] = key_active
            v_cache[:, active_idx, :, :] = value_active
            cache_event = "update"

        self.k_cache[cache_id] = k_cache.detach()
        self.v_cache[cache_id] = v_cache.detach()
        self._trace_layer(layer_key, int(key_active.shape[1]), int(total_tokens), cache_event)
        return k_cache, v_cache

    def _has_kv_cache_for_all_layers(
        self,
        transformer,
        cache_key: str,
        batch_size: int,
        image_tokens: int,
        encoder_tokens: int,
        device: torch.device,
    ) -> bool:
        self._ensure_processors(transformer)
        expected_tokens = int(image_tokens) + int(encoder_tokens)
        for layer_kind, layer_idx in self._expected_layer_keys:
            k = self.k_cache.get((cache_key, layer_kind, layer_idx))
            v = self.v_cache.get((cache_key, layer_kind, layer_idx))
            if k is None or v is None:
                return False
            expected_heads, expected_head_dim = self._expected_layer_shapes[(layer_kind, layer_idx)]
            if (
                k.dim() != 4
                or v.dim() != 4
                or k.shape != v.shape
                or int(k.shape[0]) != int(batch_size)
                or int(k.shape[1]) != expected_tokens
                or int(k.shape[2]) != expected_heads
                or int(k.shape[3]) != expected_head_dim
                or k.device != v.device
                or k.device != device
            ):
                return False
        return True

    def _trace_layer(
        self,
        layer_key: tuple[str, int],
        query_tokens: int,
        full_tokens: int,
        cache_event: str,
    ) -> None:
        active_image_tokens = int(self.runtime.get("active_image_tokens", 0))
        encoder_tokens = int(self.runtime.get("encoder_tokens", 0))
        active_seq_tokens = encoder_tokens + active_image_tokens
        _trace(
            self.runtime.get("step_idx"),
            str(self.runtime.get("trace_label", self.runtime.get("cache_key", "default"))),
            self.runtime.get("step_phase") or self.runtime.get("forward_mode") or "unknown",
            (
                f"[{layer_key[0]}:{layer_key[1]:02d}] mode={self.runtime.get('forward_mode', 'unknown')} "
                f"active_seq={active_seq_tokens}/{full_tokens} q={query_tokens} kv={full_tokens} cache={cache_event}"
            ),
        )


def _run_flux2k_branch(
    *,
    pipe: Flux2KleinPipeline,
    engine: Flux2KAttentionCompareEngine,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    latent_ids: torch.Tensor,
    step_plan,
    cache_key: str,
    step_idx: int,
    step_phase: str,
    capture_label: str | None,
    guidance_scale: float,
    negative_prompt_embeds: torch.Tensor | None,
    negative_text_ids: torch.Tensor | None,
    frozen_cache_mode: str = "rolling",
) -> tuple[torch.Tensor, dict | None]:
    with pipe.transformer.cache_context("cond"):
        noise_pred = engine.forward(
            transformer=pipe.transformer,
            hidden_states=latents.to(pipe.transformer.dtype),
            timestep=timestep / 1000,
            guidance=None,
            pooled_projections=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            joint_attention_kwargs=None,
            step_plan=step_plan,
            cache_key=cache_key,
            step_idx=step_idx,
            step_phase=step_phase,
            trace_label="cond",
            capture_label=capture_label,
            frozen_cache_mode=frozen_cache_mode,
        )
    cond_capture = engine.pop_last_capture()

    if negative_prompt_embeds is not None and negative_text_ids is not None:
        with pipe.transformer.cache_context("uncond"):
            neg_noise_pred = engine.forward(
                transformer=pipe.transformer,
                hidden_states=latents.to(pipe.transformer.dtype),
                timestep=timestep / 1000,
                guidance=None,
                pooled_projections=None,
                encoder_hidden_states=negative_prompt_embeds,
                txt_ids=negative_text_ids,
                img_ids=latent_ids,
                joint_attention_kwargs=None,
                step_plan=step_plan,
                cache_key=f"{cache_key}_uncond",
                step_idx=step_idx,
                step_phase=step_phase,
                trace_label="uncond",
                capture_label=None,
                frozen_cache_mode=frozen_cache_mode,
            )
        noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

    return noise_pred, cond_capture


def _compare_attention_capture(
    *,
    scenario_name: str,
    step_idx: int,
    text_tokens: int,
    active_token_indices: torch.Tensor,
    baseline_capture: dict | None,
    candidate_capture: dict | None,
) -> list[dict]:
    if baseline_capture is None or candidate_capture is None:
        return []

    seq_active_idx = active_token_indices.long()
    if text_tokens > 0:
        text_idx = torch.arange(text_tokens, device=seq_active_idx.device, dtype=seq_active_idx.dtype)
        seq_active_idx = torch.cat([text_idx, seq_active_idx + text_tokens], dim=0)

    records: list[dict] = []
    baseline_layers = baseline_capture.get("layers", {})
    candidate_layers = candidate_capture.get("layers", {})

    for layer_key in sorted(set(baseline_layers.keys()) & set(candidate_layers.keys())):
        baseline_output = baseline_layers[layer_key].get("attn_output")
        candidate_output = candidate_layers[layer_key].get("attn_output")
        if baseline_output is None or candidate_output is None:
            continue

        ref = baseline_output.index_select(dim=1, index=seq_active_idx.to(baseline_output.device))
        diff = candidate_output.float() - ref.float()
        abs_diff = diff.abs()

        record = {
            "scenario": scenario_name,
            "step_idx": int(step_idx),
            "layer_kind": layer_key[0],
            "layer_idx": int(layer_key[1]),
            "tokens_compared": int(candidate_output.shape[1]),
            "hidden_dim": int(candidate_output.shape[2]),
            "mean_abs": float(abs_diff.mean().item()),
            "max_abs": float(abs_diff.max().item()),
            "rmse": float(torch.sqrt(torch.mean(diff.square())).item()),
        }
        print(
            f"[compare][step={step_idx:02d}][{scenario_name}][{layer_key[0]}:{layer_key[1]:02d}] "
            f"tokens={record['tokens_compared']} mean_abs={record['mean_abs']:.3e} "
            f"max_abs={record['max_abs']:.3e} rmse={record['rmse']:.3e}",
            flush=True,
        )
        records.append(record)

    return records


def _clone_scheduler_like(pipe: Flux2KleinPipeline, num_inference_steps: int, device, sigmas, mu):
    scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
    retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(0)
    return scheduler


def _plot_metric_curves(records: list[dict], metric: str, output_path: Path) -> None:
    steps = sorted({int(record["step_idx"]) for record in records})
    layer_kinds = ["joint", "single"]
    scenarios = ["selective_no_kv", "selective_stale_kv"]

    fig, axes = plt.subplots(
        nrows=max(1, len(steps)),
        ncols=len(layer_kinds),
        figsize=(12, 4 * max(1, len(steps))),
        squeeze=False,
    )

    for row_idx, step_idx in enumerate(steps):
        for col_idx, layer_kind in enumerate(layer_kinds):
            ax = axes[row_idx][col_idx]
            for scenario in scenarios:
                rows = [
                    record for record in records
                    if int(record["step_idx"]) == step_idx
                    and record["scenario"] == scenario
                    and record["layer_kind"] == layer_kind
                ]
                rows.sort(key=lambda record: int(record["layer_idx"]))
                if not rows:
                    continue
                x = [int(record["layer_idx"]) for record in rows]
                y = [float(record[metric]) for record in rows]
                ax.plot(x, y, marker="o", linewidth=2, label=scenario.replace("selective_", ""))

            ax.set_title(f"step={step_idx} {layer_kind}")
            ax.set_xlabel("layer")
            ax.set_ylabel(metric)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_stale_ratio(records: list[dict], output_path: Path) -> None:
    steps = sorted({int(record["step_idx"]) for record in records})
    layer_kinds = ["joint", "single"]
    fig, axes = plt.subplots(
        nrows=max(1, len(steps)),
        ncols=len(layer_kinds),
        figsize=(12, 4 * max(1, len(steps))),
        squeeze=False,
    )

    for row_idx, step_idx in enumerate(steps):
        for col_idx, layer_kind in enumerate(layer_kinds):
            ax = axes[row_idx][col_idx]
            no_kv_rows = [
                record for record in records
                if int(record["step_idx"]) == step_idx
                and record["scenario"] == "selective_no_kv"
                and record["layer_kind"] == layer_kind
            ]
            stale_rows = [
                record for record in records
                if int(record["step_idx"]) == step_idx
                and record["scenario"] == "selective_stale_kv"
                and record["layer_kind"] == layer_kind
            ]
            no_kv_rows.sort(key=lambda record: int(record["layer_idx"]))
            stale_rows.sort(key=lambda record: int(record["layer_idx"]))
            if not no_kv_rows or not stale_rows:
                continue

            x = [int(record["layer_idx"]) for record in no_kv_rows]
            ratio = [
                float(stale["mean_abs"]) / max(float(no_kv["mean_abs"]), 1e-12)
                for no_kv, stale in zip(no_kv_rows, stale_rows)
            ]
            ax.plot(x, ratio, marker="o", linewidth=2, color="#d55e00")
            ax.axhline(1.0, linestyle="--", color="black", linewidth=1)
            ax.set_title(f"step={step_idx} {layer_kind} stale/no_kv")
            ax.set_xlabel("layer")
            ax.set_ylabel("mean_abs ratio")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_summary(records: list[dict], prompt: str, num_inference_steps: int, output_dir: Path) -> Path:
    plot_paths = [
        output_dir / "cubic_attention_compare_mean_abs.png",
        output_dir / "cubic_attention_compare_rmse.png",
        output_dir / "cubic_attention_compare_ratio.png",
    ]
    _plot_metric_curves(records, metric="mean_abs", output_path=plot_paths[0])
    _plot_metric_curves(records, metric="rmse", output_path=plot_paths[1])
    _plot_stale_ratio(records, output_path=plot_paths[2])

    summary = {
        "prompt": prompt,
        "steps": int(num_inference_steps),
        "records": records,
        "plots": [str(path) for path in plot_paths],
    }
    summary_path = output_dir / "cubic_attention_compare.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary_path


def run_selective_attention_comparison(
    pipe: Flux2KleinPipeline,
    prompt: str,
    height: int,
    width: int,
    guidance_scale: float,
    num_inference_steps: int,
    generator: torch.Generator,
    max_sequence_length: int,
    output_dir: Path,
) -> Path:
    with torch.inference_mode():
        cubic_cfg = SimpleNamespace(
            patch_size=(1, 1, 1),
            curvature_percentile=0.9,
            partition_gpu_aggregation=True,
            min_block_size=2,
            max_block_size=16,
            min_block_tokens=16,
            max_block_tokens=256,
            cv_threshold=0.5,
            max_blocksize_candidates=36,
            flux2k_max_cooldown_steps=2,
        )
        monitor = CurvatureMonitor(cubic_cfg)
        partitioner = AdaptivePartitioner(cubic_cfg)
        optimizer = Flux2KAdaptiveOptimizer(cubic_cfg)

        prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_embeds=None,
            device=pipe._execution_device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )

        negative_prompt_embeds = None
        negative_text_ids = None
        if guidance_scale > 1.0 and not pipe.config.is_distilled:
            negative_prompt_embeds, negative_text_ids = pipe.encode_prompt(
                prompt="",
                prompt_embeds=None,
                device=pipe._execution_device,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
            )

        num_channels_latents = pipe.transformer.config.in_channels // 4
        latents_init, latent_ids = pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=pipe._execution_device,
            generator=generator,
            latents=None,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if getattr(pipe.scheduler.config, "use_flow_sigmas", False):
            sigmas = None

        image_seq_len = latents_init.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler,
            num_inference_steps,
            pipe._execution_device,
            sigmas=sigmas,
            mu=mu,
        )
        if hasattr(pipe.scheduler, "set_begin_index"):
            pipe.scheduler.set_begin_index(0)

        schedulers = {
            "baseline_full": _clone_scheduler_like(pipe, num_inference_steps, pipe._execution_device, sigmas, mu),
            "selective_no_kv": _clone_scheduler_like(pipe, num_inference_steps, pipe._execution_device, sigmas, mu),
            "selective_stale_kv": _clone_scheduler_like(pipe, num_inference_steps, pipe._execution_device, sigmas, mu),
        }

        baseline_engine = Flux2KAttentionCompareEngine(use_kv_cache=False)
        selective_no_kv_engine = Flux2KAttentionCompareEngine(use_kv_cache=False)
        selective_stale_kv_engine = Flux2KAttentionCompareEngine(use_kv_cache=True)
        baseline_engine.reset()
        selective_no_kv_engine.reset()
        selective_stale_kv_engine.reset()

        latents_by_scenario = {
            "baseline_full": latents_init.clone(),
            "selective_no_kv": latents_init.clone(),
            "selective_stale_kv": latents_init.clone(),
        }

        current_partition = None
        current_policy_plan = None
        records: list[dict] = []
        anchor_steps = 2

        for i, t in enumerate(timesteps):
            timestep = t.expand(latents_init.shape[0]).to(latents_init.dtype)
            phase = Phase.CRUISE_ANCHOR if i < anchor_steps else Phase.CRUISE_SELECTIVE
            phase_label = phase.name.lower().replace("cruise_", "")
            step_plan = None

            if phase == Phase.CRUISE_SELECTIVE and current_partition is not None and current_policy_plan is not None:
                step_plan = optimizer.build_step_plan(
                    partition=current_partition,
                    plan=current_policy_plan,
                    step_idx=i,
                )
                if step_plan is None:
                    phase = Phase.COOLDOWN
                    phase_label = phase.name.lower()

            print(f"[compare][step={i:02d}][{phase_label}] begin", flush=True)

            baseline_noise_pred, baseline_capture = _run_flux2k_branch(
                pipe=pipe,
                engine=baseline_engine,
                latents=latents_by_scenario["baseline_full"],
                timestep=timestep,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                latent_ids=latent_ids,
                step_plan=None,
                cache_key="cmp_baseline",
                step_idx=i,
                step_phase=phase_label,
                capture_label="baseline_full" if step_plan is not None else None,
                guidance_scale=guidance_scale,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_text_ids=negative_text_ids,
            )

            baseline_unpacked = pipe._unpack_latents_with_ids(latents_by_scenario["baseline_full"], latent_ids)
            baseline_noise_unpacked = pipe._unpack_latents_with_ids(baseline_noise_pred, latent_ids)
            if phase == Phase.CRUISE_ANCHOR:
                curvature_map = monitor.update(
                    x=baseline_unpacked.unsqueeze(2),
                    v=baseline_noise_unpacked.unsqueeze(2),
                    t=_normalize_t_to_unit_interval(t, pipe.scheduler),
                    step_idx=i,
                )
                if curvature_map is not None:
                    ft, ht, wt = curvature_map.shape
                    current_partition = partitioner.partition(Ft=ft, Ht=ht, Wt=wt, curvature_map=curvature_map)
                    block_curvatures = monitor.get_block_curvatures(current_partition, curvature_map=curvature_map)
                    current_policy_plan = optimizer.optimize(
                        partition=current_partition,
                        block_curvatures=block_curvatures,
                        total_steps=len(timesteps),
                        anchor_steps=anchor_steps,
                    )

            local_noise_pred, local_capture = _run_flux2k_branch(
                pipe=pipe,
                engine=selective_no_kv_engine,
                latents=latents_by_scenario["selective_no_kv"],
                timestep=timestep,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                latent_ids=latent_ids,
                step_plan=step_plan,
                cache_key="cmp_local",
                step_idx=i,
                step_phase=phase_label,
                capture_label="selective_no_kv" if step_plan is not None else None,
                guidance_scale=guidance_scale,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_text_ids=negative_text_ids,
            )

            stale_noise_pred, stale_capture = _run_flux2k_branch(
                pipe=pipe,
                engine=selective_stale_kv_engine,
                latents=latents_by_scenario["selective_stale_kv"],
                timestep=timestep,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                latent_ids=latent_ids,
                step_plan=step_plan,
                cache_key="cmp_stale",
                step_idx=i,
                step_phase=phase_label,
                capture_label="selective_stale_kv" if step_plan is not None else None,
                guidance_scale=guidance_scale,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_text_ids=negative_text_ids,
                frozen_cache_mode="anchor_base" if step_plan is not None else "rolling",
            )

            if i == anchor_steps - 1:
                selective_stale_kv_engine.snapshot_anchor_cache()

            if step_plan is not None:
                active_idx = step_plan.active_token_indices.to(latent_ids.device)
                records.extend(
                    _compare_attention_capture(
                        scenario_name="selective_no_kv",
                        step_idx=i,
                        text_tokens=int(prompt_embeds.shape[1]),
                        active_token_indices=active_idx,
                        baseline_capture=baseline_capture,
                        candidate_capture=local_capture,
                    )
                )
                records.extend(
                    _compare_attention_capture(
                        scenario_name="selective_stale_kv",
                        step_idx=i,
                        text_tokens=int(prompt_embeds.shape[1]),
                        active_token_indices=active_idx,
                        baseline_capture=baseline_capture,
                        candidate_capture=stale_capture,
                    )
                )

            latents_by_scenario["baseline_full"] = schedulers["baseline_full"].step(
                baseline_noise_pred, t, latents_by_scenario["baseline_full"], return_dict=False
            )[0]
            latents_by_scenario["selective_no_kv"] = schedulers["selective_no_kv"].step(
                local_noise_pred, t, latents_by_scenario["selective_no_kv"], return_dict=False
            )[0]
            latents_by_scenario["selective_stale_kv"] = schedulers["selective_stale_kv"].step(
                stale_noise_pred, t, latents_by_scenario["selective_stale_kv"], return_dict=False
            )[0]

        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = _write_summary(records, prompt, num_inference_steps, output_dir)
        print(f"Attention comparison saved to: {summary_path.resolve()}", flush=True)
        return summary_path


def _slice_img_ids(img_ids: torch.Tensor | None, active_idx: torch.Tensor) -> torch.Tensor | None:
    if img_ids is None:
        return None
    if img_ids.dim() == 3:
        return img_ids.index_select(dim=1, index=active_idx.to(img_ids.device))
    if img_ids.dim() == 2:
        return img_ids.index_select(dim=0, index=active_idx.to(img_ids.device))
    raise ValueError(f"Unsupported img_ids shape for selective slicing: {tuple(img_ids.shape)}")


def _normalize_t_to_unit_interval(t: torch.Tensor | float, scheduler) -> float:
    if isinstance(t, torch.Tensor):
        if t.numel() == 0:
            return 0.0
        t_val = float(t.detach().reshape(-1)[0].item())
    else:
        t_val = float(t)

    if t_val > 1.0:
        scheduler_steps = float(getattr(scheduler.config, "num_train_timesteps", 1000) - 1)
        t_val = t_val / max(1.0, scheduler_steps)
    return max(0.0, min(1.0, t_val))


def _trace(step_idx: int | None, trace_label: str, phase: str, message: str) -> None:
    step_part = "na" if step_idx is None else f"{int(step_idx):02d}"
    print(f"[cubic][step={step_part}][{trace_label}][{phase}] {message}", flush=True)
