from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch

from diffusers.models.attention_dispatch import dispatch_attention_fn

from firered.transformer import apply_rotary_emb_qwen


logger = logging.getLogger(__name__)


class FireRedCubicKVCacheAttnProcessor:
    """FireRed joint attention with active image Q and cached full image K/V."""

    def __init__(self, engine: "FireRedSelectiveForwardEngine", layer_kind: str, layer_idx: int):
        self.engine = engine
        self.layer_kind = layer_kind
        self.layer_idx = layer_idx
        self._attention_backend = None
        self._parallel_config = None

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("FireRedCubicKVCacheAttnProcessor requires encoder_hidden_states.")

        runtime = self.engine._runtime
        seq_txt = encoder_hidden_states.shape[1]

        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        if runtime is not None:
            img_key, img_value = self.engine._update_and_get_image_kv_cache(
                cache_key=runtime["cache_key"],
                layer_key=(self.layer_kind, self.layer_idx),
                key_active=img_key,
                value_active=img_value,
                active_idx=runtime["active_idx"],
                total_tokens=runtime["total_tokens"],
            )
            attention_mask = self._expand_attention_mask_for_full_image_kv(
                attention_mask=attention_mask,
                text_tokens=seq_txt,
                total_image_tokens=int(runtime["total_tokens"]),
                batch_size=int(img_query.shape[0]),
                device=img_query.device,
            )

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)

        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())
        return img_attn_output, txt_attn_output

    @staticmethod
    def _expand_attention_mask_for_full_image_kv(
        attention_mask: torch.Tensor | None,
        text_tokens: int,
        total_image_tokens: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.ndim != 2:
            return attention_mask

        text_mask = attention_mask[:, :text_tokens].to(device=device)
        if int(text_mask.shape[0]) != batch_size:
            return attention_mask
        image_mask = torch.ones(
            (batch_size, total_image_tokens),
            dtype=text_mask.dtype,
            device=device,
        )
        return torch.cat([text_mask, image_mask], dim=1)


class FireRedSelectiveForwardEngine:
    """Selective forward for FireRed.

    Strategy:
    - main-image tokens follow the Cubic step plan
    - condition-image tokens are either fully refreshed or fully reused
    - active image tokens compute Q/K/V and update per-layer image KV cache
    - frozen image tokens reuse cached K/V for active-token attention
    - frozen image outputs are reused from the previous full/selective step
    """

    def __init__(self, config):
        self.config = config
        self.output_cache: Dict[str, torch.Tensor] = {}
        self.k_cache: Dict[Tuple[str, str, int], torch.Tensor] = {}
        self.v_cache: Dict[Tuple[str, str, int], torch.Tensor] = {}
        self._runtime: Optional[dict] = None
        self._patched_transformer_id: Optional[int] = None
        self._expected_layer_keys: list[Tuple[str, int]] = []

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
        encoder_hidden_states_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        img_shapes,
        attention_kwargs,
        step_plan=None,
        cond_step_plan=None,
        cache_key: str = "default",
        main_token_count: int | None = None,
        step_idx: int | None = None,
        step_phase: str | None = None,
    ) -> torch.Tensor:
        seq_len = int(hidden_states.shape[1])
        main_token_count = seq_len if main_token_count is None else int(main_token_count)

        if main_token_count < 0 or main_token_count > seq_len:
            raise ValueError(f"Invalid main_token_count={main_token_count} for hidden_states seq_len={seq_len}.")

        main_use_selective = step_plan is not None and (not step_plan.is_full_compute)
        cond_is_planned = cond_step_plan is not None
        cond_use_selective = cond_is_planned and (not cond_step_plan.is_full_compute)
        use_selective = main_use_selective or cond_use_selective
        if not use_selective:
            return self._full_forward_and_refresh_cache(
                transformer=transformer,
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                encoder_hidden_states=encoder_hidden_states,
                img_shapes=img_shapes,
                attention_kwargs=attention_kwargs,
                cache_key=cache_key,
                step_idx=step_idx,
                step_phase=step_phase,
                reason="no_selective_plan",
            )

        if main_use_selective:
            active_main_idx = torch.unique(step_plan.active_token_indices.long().to(hidden_states.device), sorted=True)
        else:
            active_main_idx = torch.empty(0, device=hidden_states.device, dtype=torch.long)

        if active_main_idx.numel() > 0 and int(active_main_idx.max().item()) >= main_token_count:
            logger.info(
                "[firered-selective] step=%s branch=%s active index overflow (max=%d, main=%d), fallback full.",
                "na" if step_idx is None else str(int(step_idx)),
                cache_key,
                int(active_main_idx.max().item()),
                main_token_count,
            )
            return self._full_forward_and_refresh_cache(
                transformer=transformer,
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                encoder_hidden_states=encoder_hidden_states,
                img_shapes=img_shapes,
                attention_kwargs=attention_kwargs,
                cache_key=cache_key,
                step_idx=step_idx,
                step_phase=step_phase,
                reason="step_plan_overflow",
            )

        cached = self.output_cache.get(cache_key)
        if cached is None or cached.shape != hidden_states.shape:
            return self._full_forward_and_refresh_cache(
                transformer=transformer,
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                encoder_hidden_states=encoder_hidden_states,
                img_shapes=img_shapes,
                attention_kwargs=attention_kwargs,
                cache_key=cache_key,
                step_idx=step_idx,
                step_phase=step_phase,
                reason="output_cache_miss",
            )

        cond_token_count = max(0, seq_len - main_token_count)
        if cond_is_planned and cond_token_count > 0 and cond_step_plan.is_full_compute:
            condition_idx = torch.arange(
                main_token_count,
                seq_len,
                device=hidden_states.device,
                dtype=torch.long,
            )
        elif cond_use_selective and cond_token_count > 0:
            cond_local_idx = torch.unique(cond_step_plan.active_token_indices.long().to(hidden_states.device), sorted=True)
            if cond_local_idx.numel() > 0 and int(cond_local_idx.max().item()) >= cond_token_count:
                logger.info(
                    "[firered-selective] step=%s branch=%s cond index overflow (max=%d, cond=%d), fallback full.",
                    "na" if step_idx is None else str(int(step_idx)),
                    cache_key,
                    int(cond_local_idx.max().item()),
                    cond_token_count,
                )
                return self._full_forward_and_refresh_cache(
                    transformer=transformer,
                    hidden_states=hidden_states,
                    timestep=timestep,
                    guidance=guidance,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    img_shapes=img_shapes,
                    attention_kwargs=attention_kwargs,
                    cache_key=cache_key,
                    step_idx=step_idx,
                    step_phase=step_phase,
                    reason="cond_step_plan_overflow",
                )
            condition_idx = cond_local_idx + main_token_count
        else:
            condition_idx = torch.empty(0, device=hidden_states.device, dtype=torch.long)

        selected_idx = torch.unique(torch.cat([active_main_idx, condition_idx], dim=0), sorted=True)

        if selected_idx.numel() == 0:
            logger.info(
                "[firered-selective] step=%s branch=%s phase=%s selected=0 -> reuse cached output.",
                "na" if step_idx is None else str(int(step_idx)),
                cache_key,
                step_phase or "selective",
            )
            return cached.clone()

        if not self._has_kv_cache_for_all_layers(transformer, cache_key=cache_key, total_tokens=seq_len):
            return self._full_forward_and_refresh_cache(
                transformer=transformer,
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                encoder_hidden_states=encoder_hidden_states,
                img_shapes=img_shapes,
                attention_kwargs=attention_kwargs,
                cache_key=cache_key,
                step_idx=step_idx,
                step_phase=step_phase,
                reason="kv_cache_miss",
            )

        if int(selected_idx.numel()) >= seq_len:
            return self._full_forward_and_refresh_cache(
                transformer=transformer,
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                encoder_hidden_states=encoder_hidden_states,
                img_shapes=img_shapes,
                attention_kwargs=attention_kwargs,
                cache_key=cache_key,
                step_idx=step_idx,
                step_phase=step_phase,
                reason="all_tokens_selected",
            )

        logger.info(
            "[firered-selective] step=%s branch=%s phase=%s main_active=%d/%d cond_active=%d total_selected=%d/%d",
            "na" if step_idx is None else str(int(step_idx)),
            cache_key,
            step_phase or "selective",
            int(active_main_idx.numel()),
            main_token_count,
            int(condition_idx.numel()),
            int(selected_idx.numel()),
            seq_len,
        )

        active_hidden_states = hidden_states.index_select(dim=1, index=selected_idx)
        active_output = self._run_transformer_with_kv_cache(
            transformer=transformer,
            hidden_states=active_hidden_states,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            encoder_hidden_states=encoder_hidden_states,
            img_shapes=img_shapes,
            attention_kwargs=attention_kwargs,
            image_token_indices=selected_idx,
            cache_key=cache_key,
            active_idx=selected_idx,
            total_tokens=seq_len,
        )

        output = cached.clone()
        output[:, selected_idx, :] = active_output
        self.output_cache[cache_key] = output.detach()
        return output

    def _full_forward_and_refresh_cache(
        self,
        transformer,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance,
        encoder_hidden_states_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        img_shapes,
        attention_kwargs,
        cache_key: str,
        step_idx: int | None,
        step_phase: str | None,
        reason: str,
    ) -> torch.Tensor:
        logger.info(
            "[firered-selective] step=%s branch=%s phase=%s mode=full reason=%s",
            "na" if step_idx is None else str(int(step_idx)),
            cache_key,
            step_phase or "full",
            reason,
        )
        seq_len = int(hidden_states.shape[1])
        all_idx = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long)
        output = self._run_transformer_with_kv_cache(
            transformer=transformer,
            hidden_states=hidden_states,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            encoder_hidden_states=encoder_hidden_states,
            img_shapes=img_shapes,
            attention_kwargs=attention_kwargs,
            image_token_indices=None,
            cache_key=cache_key,
            active_idx=all_idx,
            total_tokens=seq_len,
        )
        self.output_cache[cache_key] = output.detach()
        return output

    def _ensure_cubic_attn_processors(self, transformer) -> None:
        transformer_id = id(transformer)
        if self._patched_transformer_id == transformer_id:
            return

        self._expected_layer_keys = []
        for idx, block in enumerate(transformer.transformer_blocks):
            block.attn.set_processor(FireRedCubicKVCacheAttnProcessor(self, layer_kind="joint", layer_idx=idx))
            self._expected_layer_keys.append(("joint", idx))

        self._patched_transformer_id = transformer_id

    def _run_transformer_with_kv_cache(
        self,
        transformer,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance,
        encoder_hidden_states_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        img_shapes,
        attention_kwargs,
        image_token_indices: torch.Tensor | None,
        cache_key: str,
        active_idx: torch.Tensor,
        total_tokens: int,
    ) -> torch.Tensor:
        self._ensure_cubic_attn_processors(transformer)
        self._runtime = {
            "cache_key": cache_key,
            "active_idx": active_idx,
            "total_tokens": int(total_tokens),
        }
        try:
            return transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                encoder_hidden_states=encoder_hidden_states,
                img_shapes=img_shapes,
                attention_kwargs=attention_kwargs,
                image_token_indices=image_token_indices,
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
        cache_id = (cache_key, layer_key[0], layer_key[1])
        k_cache = self.k_cache.get(cache_id)
        v_cache = self.v_cache.get(cache_id)

        cache_shape_ok = (
            k_cache is not None
            and v_cache is not None
            and k_cache.shape[0] == key_active.shape[0]
            and k_cache.shape[1] == total_tokens
            and k_cache.shape[2] == key_active.shape[2]
            and k_cache.shape[3] == key_active.shape[3]
            and v_cache.shape[0] == value_active.shape[0]
            and v_cache.shape[1] == total_tokens
            and v_cache.shape[2] == value_active.shape[2]
            and v_cache.shape[3] == value_active.shape[3]
        )
        if not cache_shape_ok:
            k_cache = torch.zeros(
                key_active.shape[0],
                total_tokens,
                key_active.shape[2],
                key_active.shape[3],
                dtype=key_active.dtype,
                device=key_active.device,
            )
            v_cache = torch.zeros(
                value_active.shape[0],
                total_tokens,
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

        self.k_cache[cache_id] = k_cache.detach()
        self.v_cache[cache_id] = v_cache.detach()
        return k_cache, v_cache

    def _has_kv_cache_for_all_layers(self, transformer, cache_key: str, total_tokens: int) -> bool:
        self._ensure_cubic_attn_processors(transformer)
        for layer_kind, layer_idx in self._expected_layer_keys:
            k = self.k_cache.get((cache_key, layer_kind, layer_idx))
            v = self.v_cache.get((cache_key, layer_kind, layer_idx))
            if k is None or v is None:
                return False
            if int(k.shape[1]) != int(total_tokens) or int(v.shape[1]) != int(total_tokens):
                return False
        return True
