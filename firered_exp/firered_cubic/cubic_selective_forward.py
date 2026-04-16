from __future__ import annotations

import logging
from typing import Dict

import torch


logger = logging.getLogger(__name__)


class FireRedSelectiveForwardEngine:
    """Selective forward for FireRed.

    Current minimal strategy:
    - main-image tokens follow the Cubic step plan
    - condition-image tokens follow an optional Cubic cond-step plan
    - when cond-step plan is absent, condition-image tokens stay frozen on selective steps
    - condition-image tokens are refreshed on full/anchor steps
    - frozen main-image outputs are reused from the previous full/selective step
    """

    def __init__(self, config):
        self.config = config
        self.output_cache: Dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        self.output_cache.clear()

    @staticmethod
    def _is_cond_active_in_selective(step_phase: str | None) -> bool:
        # Default policy: cond KV is frozen in cruise-selective steps and refreshed on anchor/full.
        return False

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
        cond_use_selective = cond_step_plan is not None and (not cond_step_plan.is_full_compute)
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
        if cond_use_selective and cond_token_count > 0:
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
            cond_active_in_selective = self._is_cond_active_in_selective(step_phase)
            if cond_active_in_selective:
                condition_idx = torch.arange(main_token_count, seq_len, device=hidden_states.device, dtype=torch.long)
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
        active_output = self._run_transformer(
            transformer=transformer,
            hidden_states=active_hidden_states,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            encoder_hidden_states=encoder_hidden_states,
            img_shapes=img_shapes,
            attention_kwargs=attention_kwargs,
            image_token_indices=selected_idx,
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
        output = self._run_transformer(
            transformer=transformer,
            hidden_states=hidden_states,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            encoder_hidden_states=encoder_hidden_states,
            img_shapes=img_shapes,
            attention_kwargs=attention_kwargs,
            image_token_indices=None,
        )
        self.output_cache[cache_key] = output.detach()
        return output

    @staticmethod
    def _run_transformer(
        transformer,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance,
        encoder_hidden_states_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        img_shapes,
        attention_kwargs,
        image_token_indices: torch.Tensor | None,
    ) -> torch.Tensor:
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
