"""Pure selective forward engine for Flux2K cubic."""

from __future__ import annotations

from typing import Dict

import torch


class Flux2KSelectiveForwardEngine:
    """Token-selective forward with cached hidden-state reuse.

    The main cubic runtime keeps a single behavior:
    compute active image tokens for selective steps and reuse the previous
    hidden states for frozen image tokens.
    """

    def __init__(self, config):
        self.config = config
        self.output_cache: Dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        self.output_cache.clear()

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
                reason="no_selective_plan",
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
                    reason="empty_active_set_without_output_cache",
                )

            self._trace(
                step_idx=step_idx,
                trace_label=branch_label,
                phase=step_phase or "selective",
                message=f"forward mode=selective img_active=0/{seq_len} txt={text_tokens} cache=reuse",
            )
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
                reason=reason,
            )

        active_hidden = hidden_states.index_select(dim=1, index=active_idx)
        selected_img_ids = self._slice_img_ids(img_ids, active_idx)
        self._trace(
            step_idx=step_idx,
            trace_label=branch_label,
            phase=step_phase or "selective",
            message=f"forward mode=selective img_active={active_count}/{seq_len} txt={text_tokens} cache=hit",
        )

        active_out = self._run_transformer(
            transformer=transformer,
            hidden_states=active_hidden,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=selected_img_ids,
            joint_attention_kwargs=joint_attention_kwargs,
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
        reason: str,
    ) -> torch.Tensor:
        seq_len = int(hidden_states.shape[1])
        text_tokens = int(encoder_hidden_states.shape[1])
        self._trace(
            step_idx=step_idx,
            trace_label=trace_label,
            phase=step_phase or "full",
            message=f"forward mode=full img_active={seq_len}/{seq_len} txt={text_tokens} reason={reason}",
        )
        output = self._run_transformer(
            transformer=transformer,
            hidden_states=hidden_states,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        self.output_cache[cache_key] = output.detach()
        return output

    @staticmethod
    def _run_transformer(
        transformer,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor | None,
        joint_attention_kwargs,
    ) -> torch.Tensor:
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

    @staticmethod
    def _slice_img_ids(img_ids: torch.Tensor | None, active_idx: torch.Tensor) -> torch.Tensor | None:
        if img_ids is None:
            return None
        if img_ids.dim() == 3:
            return img_ids.index_select(dim=1, index=active_idx.to(img_ids.device))
        if img_ids.dim() == 2:
            return img_ids.index_select(dim=0, index=active_idx.to(img_ids.device))
        raise ValueError(f"Unsupported img_ids shape for selective slicing: {tuple(img_ids.shape)}")

    @staticmethod
    def _trace(
        step_idx: int | None,
        trace_label: str,
        phase: str,
        message: str,
    ) -> None:
        step_part = "na" if step_idx is None else f"{int(step_idx):02d}"
        print(
            f"[cubic][step={step_part}][{trace_label}][{phase}] {message}",
            flush=True,
        )
