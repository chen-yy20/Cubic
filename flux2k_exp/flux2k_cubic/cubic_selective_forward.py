"""Selective forward engine for flux2k cubic.

Supports two modes:
- pure selective (no KV cache)
- selective with per-layer frozen KV cache
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from .attn import Flux2CubicKVCacheAttnProcessor, Flux2CubicKVCacheParallelSelfAttnProcessor


class Flux2KSelectiveForwardEngine:
    """Token-selective forward with optional per-layer KV cache."""

    def __init__(self, config):
        self.config = config
        self.output_cache: Dict[str, torch.Tensor] = {}
        self.k_cache: Dict[Tuple[str, str, int], torch.Tensor] = {}
        self.v_cache: Dict[Tuple[str, str, int], torch.Tensor] = {}
        self._runtime: Optional[dict] = None
        self._patched_transformer_id: Optional[int] = None
        self._expected_layer_keys: list[Tuple[str, int]] = []
        self.use_kv_cache = False

    def set_use_kv_cache(self, enabled: bool) -> None:
        self.use_kv_cache = bool(enabled)

    def reset(self) -> None:
        self.output_cache.clear()
        self.k_cache.clear()
        self.v_cache.clear()
        self._runtime = None

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
    ) -> torch.Tensor:
        del pooled_projections

        use_selective = step_plan is not None and (not step_plan.is_full_compute)

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
            )

        batch_size, seq_len, _ = hidden_states.shape
        active_idx = step_plan.active_token_indices.long().to(hidden_states.device)
        active_idx = torch.unique(active_idx, sorted=True)

        cached = self.output_cache.get(cache_key)
        if active_idx.numel() == 0:
            # 全冻结步：直接复用上一轮输出缓存。
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
                )
            return cached.clone()

        if int(active_idx.numel()) >= int(seq_len):
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
            )

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
            )

        if self.use_kv_cache and not self._has_kv_cache_for_all_layers(
            transformer=transformer,
            cache_key=cache_key,
            image_tokens=seq_len,
            encoder_tokens=int(encoder_hidden_states.shape[1]),
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
            )

        active_hidden = hidden_states.index_select(dim=1, index=active_idx)
        active_img_ids = self._slice_img_ids(img_ids, active_idx)

        if self.use_kv_cache:
            active_out = self._run_transformer_with_kv_cache(
                transformer=transformer,
                hidden_states=active_hidden,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=active_img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                cache_key=cache_key,
                active_idx=active_idx,
                total_tokens=seq_len,
                encoder_tokens=int(encoder_hidden_states.shape[1]),
            )
        else:
            active_out = transformer(
                hidden_states=active_hidden,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=active_img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=False,
            )[0]

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
    ) -> torch.Tensor:
        if self.use_kv_cache:
            seq_len = int(hidden_states.shape[1])
            all_idx = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long)
            output = self._run_transformer_with_kv_cache(
                transformer=transformer,
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                cache_key=cache_key,
                active_idx=all_idx,
                total_tokens=seq_len,
                encoder_tokens=int(encoder_hidden_states.shape[1]),
            )
        else:
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

        self.output_cache[cache_key] = output.detach()
        return output

    def _ensure_cubic_attn_processors(self, transformer) -> None:
        transformer_id = id(transformer)
        if self._patched_transformer_id == transformer_id:
            return

        self._expected_layer_keys = []
        for idx, block in enumerate(transformer.transformer_blocks):
            block.attn.set_processor(Flux2CubicKVCacheAttnProcessor(self, layer_kind="joint", layer_idx=idx))
            self._expected_layer_keys.append(("joint", idx))

        for idx, block in enumerate(transformer.single_transformer_blocks):
            block.attn.set_processor(
                Flux2CubicKVCacheParallelSelfAttnProcessor(self, layer_kind="single", layer_idx=idx)
            )
            self._expected_layer_keys.append(("single", idx))

        self._patched_transformer_id = transformer_id

    def _run_transformer_with_kv_cache(
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
    ) -> torch.Tensor:
        self._ensure_cubic_attn_processors(transformer)
        self._runtime = {
            "cache_key": cache_key,
            "active_idx": active_idx,
            "total_tokens": int(total_tokens),
            "encoder_tokens": int(encoder_tokens),
        }
        try:
            return transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=False,
            )[0]
        finally:
            self._runtime = None

    def _update_and_get_image_kv_cache(
        self,
        cache_key: str,
        layer_key: Tuple[str, int],
        key_active: torch.Tensor,
        value_active: torch.Tensor,
        active_idx: torch.Tensor,
        total_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_k_key = (cache_key, layer_key[0], layer_key[1])
        cache_v_key = (cache_key, layer_key[0], layer_key[1])

        k_cache = self.k_cache.get(cache_k_key)
        v_cache = self.v_cache.get(cache_v_key)

        cache_shape_ok = (
            k_cache is not None
            and v_cache is not None
            and k_cache.shape[0] == key_active.shape[0]
            and k_cache.shape[1] == int(total_tokens)
            and k_cache.shape[2] == key_active.shape[2]
            and k_cache.shape[3] == key_active.shape[3]
            and v_cache.shape[0] == value_active.shape[0]
            and v_cache.shape[1] == int(total_tokens)
            and v_cache.shape[2] == value_active.shape[2]
            and v_cache.shape[3] == value_active.shape[3]
        )

        if not cache_shape_ok:
            k_cache = torch.zeros(
                key_active.shape[0],
                int(total_tokens),
                key_active.shape[2],
                key_active.shape[3],
                dtype=key_active.dtype,
                device=key_active.device,
            )
            v_cache = torch.zeros(
                value_active.shape[0],
                int(total_tokens),
                value_active.shape[2],
                value_active.shape[3],
                dtype=value_active.dtype,
                device=value_active.device,
            )
        else:
            k_cache = k_cache.to(device=key_active.device, dtype=key_active.dtype)
            v_cache = v_cache.to(device=value_active.device, dtype=value_active.dtype)

        k_cache[:, active_idx, :, :] = key_active
        v_cache[:, active_idx, :, :] = value_active

        self.k_cache[cache_k_key] = k_cache.detach()
        self.v_cache[cache_v_key] = v_cache.detach()
        return k_cache, v_cache

    def _update_and_get_full_kv_cache(
        self,
        cache_key: str,
        layer_key: Tuple[str, int],
        key_active: torch.Tensor,
        value_active: torch.Tensor,
        active_idx: torch.Tensor,
        total_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._update_and_get_image_kv_cache(
            cache_key=cache_key,
            layer_key=layer_key,
            key_active=key_active,
            value_active=value_active,
            active_idx=active_idx,
            total_tokens=total_tokens,
        )

    def _has_kv_cache_for_all_layers(
        self,
        transformer,
        cache_key: str,
        image_tokens: int,
        encoder_tokens: int,
    ) -> bool:
        self._ensure_cubic_attn_processors(transformer)
        for layer_kind, layer_idx in self._expected_layer_keys:
            k = self.k_cache.get((cache_key, layer_kind, layer_idx))
            v = self.v_cache.get((cache_key, layer_kind, layer_idx))
            if k is None or v is None:
                return False
            expected_tokens = int(image_tokens)
            if layer_kind == "single":
                expected_tokens = int(image_tokens) + int(encoder_tokens)
            if int(k.shape[1]) != expected_tokens or int(v.shape[1]) != expected_tokens:
                print(f"KV cache shape mismatch for layer ({layer_kind}, {layer_idx}): "
                    f"expected tokens={expected_tokens}, got k={k.shape}, v={v.shape}", flush=True)
                return False
        return True

    @staticmethod
    def _slice_img_ids(img_ids: torch.Tensor | None, active_idx: torch.Tensor) -> torch.Tensor | None:
        if img_ids is None:
            return None
        if img_ids.dim() == 3:
            return img_ids.index_select(dim=1, index=active_idx)
        if img_ids.dim() == 2:
            return img_ids.index_select(dim=0, index=active_idx)
        raise ValueError(f"Unsupported img_ids rank: {img_ids.dim()}")
