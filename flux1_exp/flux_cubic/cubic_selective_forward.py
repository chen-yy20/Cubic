from __future__ import annotations

import logging
from typing import Optional

import torch


class FluxSelectiveForwardEngine:
    """Selective forward wrapper for Flux transformer.

    Compared with WAN, Flux latent tokens are packed as a flat image-token
    sequence `[B, L, C]` with a paired `img_ids[L, 3]`. StepPlan token indices
    are applied directly on this packed sequence.
    """

    def __init__(self, config):
        self.config = config
        self.output_cache: dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        self.output_cache.clear()

    def forward(
        self,
        transformer,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor],
        pooled_projections: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: Optional[torch.Tensor],
        joint_attention_kwargs,
        step_plan=None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        use_selective = (
            step_plan is not None
            and (not step_plan.is_full_compute)
            and step_plan.active_token_indices.numel() > 0
        )

        if not use_selective:
            return self._full_forward_and_refresh_cache(
                transformer=transformer,
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                pooled_projections=pooled_projections,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                cache_key=cache_key,
            )

        batch_size, seq_len, _ = hidden_states.shape
        active_idx = step_plan.active_token_indices.long().to(hidden_states.device)
        if active_idx.numel() == 0 or int(active_idx.numel()) >= int(seq_len):
            return self._full_forward_and_refresh_cache(
                transformer=transformer,
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                pooled_projections=pooled_projections,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                cache_key=cache_key,
            )

        if int(active_idx.max().item()) >= int(seq_len):
            logging.info(
                "[flux-selective] step plan index overflow (max=%d, seq_len=%d), fallback full.",
                int(active_idx.max().item()),
                int(seq_len),
            )
            return self._full_forward_and_refresh_cache(
                transformer=transformer,
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                pooled_projections=pooled_projections,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                cache_key=cache_key,
            )

        out_cache = self.output_cache.get(cache_key)
        if out_cache is None or out_cache.shape != hidden_states.shape:
            logging.info("[flux-selective] missing/invalid output cache, fallback full.")
            return self._full_forward_and_refresh_cache(
                transformer=transformer,
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                pooled_projections=pooled_projections,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                cache_key=cache_key,
            )

        if img_ids is not None and int(img_ids.shape[0]) != int(seq_len):
            logging.info(
                "[flux-selective] img_ids/token length mismatch (%d vs %d), fallback full.",
                int(img_ids.shape[0]),
                int(seq_len),
            )
            return self._full_forward_and_refresh_cache(
                transformer=transformer,
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                pooled_projections=pooled_projections,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                cache_key=cache_key,
            )

        hidden_active = hidden_states.index_select(1, active_idx)
        img_ids_active = img_ids.index_select(0, active_idx) if img_ids is not None else None

        out_active = transformer(
            hidden_states=hidden_active,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_projections,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids_active,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]

        out_full = out_cache.to(device=hidden_states.device, dtype=out_active.dtype).clone()
        out_full[:, active_idx, :] = out_active
        self.output_cache[cache_key] = out_full.detach()

        logging.debug(
            "[flux-selective] active_tokens=%d frozen_tokens=%d batch=%d",
            int(active_idx.numel()),
            int(seq_len - active_idx.numel()),
            int(batch_size),
        )
        return out_full

    def _full_forward_and_refresh_cache(
        self,
        transformer,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor],
        pooled_projections: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: Optional[torch.Tensor],
        joint_attention_kwargs,
        cache_key: str,
    ) -> torch.Tensor:
        out = transformer(
            hidden_states=hidden_states,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_projections,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]
        self.output_cache[cache_key] = out.detach()
        return out