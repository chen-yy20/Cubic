"""Selective-phase step scheduler for Cubic.

This module converts a partition over latent/token space into a transformer-
level token plan, similar in spirit to Jano's mask transform + apply mask flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .types import Block, Partition, Phase, StepPlan


@dataclass
class StepPlanDebugInfo:
    """Human-readable summary for a generated step plan."""

    phase: Phase
    step_idx: int
    active_block_count: int
    frozen_block_count: int
    active_token_count: int
    frozen_token_count: int
    total_token_count: int
    is_full_compute: bool


class SelectiveStepScheduler:
    """Build token-level step plans from partition + per-block update intervals."""

    def __init__(self, config):
        self.config = config
        self.warmup_fraction = float(getattr(config, 'warmup_fraction', 0.15))
        self.cooldown_fraction = float(getattr(config, 'cooldown_fraction', 0.15))

    def build_step_plan(
        self,
        partition: Partition,
        block_intervals: Optional[Dict[int, int]],
        phase: Phase,
        step_idx: int,
        selective_step_idx: Optional[int] = None,
    ) -> StepPlan:
        """Generate a transformer-level StepPlan for one sampling step.

        Args:
            partition: Latent/token partition.
            block_intervals: Per-block update interval tau_b.
            phase: Inference phase.
            step_idx: Absolute sampling step.
            selective_step_idx: Phase-local step index inside cruise selective.

        Returns:
            StepPlan with active/frozen token indices.
        """
        total_tokens = partition.total_tokens
        all_token_indices = torch.arange(total_tokens, dtype=torch.long)
        if not partition.blocks:
            return StepPlan(
                phase=phase,
                active_block_ids=[],
                active_token_indices=all_token_indices,
                frozen_token_indices=torch.empty(0, dtype=torch.long),
                is_full_compute=True,
            )

        if phase != Phase.CRUISE_SELECTIVE:
            return StepPlan(
                phase=phase,
                active_block_ids=[block.block_id for block in partition.blocks],
                active_token_indices=all_token_indices,
                frozen_token_indices=torch.empty(0, dtype=torch.long),
                is_full_compute=True,
            )

        active_block_ids = self._select_active_block_ids(
            partition=partition,
            block_intervals=block_intervals or {},
            selective_step_idx=step_idx if selective_step_idx is None else selective_step_idx,
        )
        kv_cache_block_ids = [
            block_id
            for block_id in active_block_ids
            if int((block_intervals or {}).get(block_id, 1)) != 1
        ]

        token_indices_by_block = partition.get_all_token_indices()
        active_token_indices = self._concat_block_tokens(active_block_ids, token_indices_by_block)
        kv_cache_token_indices = self._concat_block_tokens(kv_cache_block_ids, token_indices_by_block)
        frozen_mask = torch.ones(total_tokens, dtype=torch.bool)
        if active_token_indices.numel() > 0:
            frozen_mask[active_token_indices] = False
        frozen_token_indices = all_token_indices[frozen_mask]

        if active_token_indices.numel() == total_tokens:
            return StepPlan(
                phase=phase,
                active_block_ids=active_block_ids,
                active_token_indices=active_token_indices,
                frozen_token_indices=torch.empty(0, dtype=torch.long),
                is_full_compute=True,
            )

        return StepPlan(
            phase=phase,
            active_block_ids=active_block_ids,
            active_token_indices=active_token_indices,
            frozen_token_indices=frozen_token_indices,
            is_full_compute=False,
            kv_cache_token_indices=kv_cache_token_indices,
        )

    def build_sequence_mask(self, step_plan: StepPlan, seq_len: Optional[int] = None) -> torch.Tensor:
        """Convert StepPlan into a transformer-style boolean sequence mask.

        True means the token position is active and should be recomputed.
        """
        if seq_len is None:
            seq_len = int(step_plan.active_token_indices.numel() + step_plan.frozen_token_indices.numel())
        mask = torch.zeros(seq_len, dtype=torch.bool)
        if step_plan.active_token_indices.numel() > 0:
            mask[step_plan.active_token_indices.long()] = True
        return mask

    def apply_sequence_mask(self, hidden_states: torch.Tensor, step_plan: StepPlan) -> torch.Tensor:
        """Select active token rows from a [B, L, C] or [L, C] tensor.

        This mirrors Jano's apply_sequence_mask behavior in spirit: build a token-
        level mask first, then gather only active positions for selective forward.
        """
        if step_plan.is_full_compute:
            return hidden_states

        indices = step_plan.active_token_indices.to(hidden_states.device)
        if hidden_states.dim() == 3:
            return hidden_states.index_select(dim=1, index=indices)
        if hidden_states.dim() == 2:
            return hidden_states.index_select(dim=0, index=indices)
        raise ValueError(f'Unsupported hidden_states rank: {hidden_states.dim()}')

    def build_token_block_map(self, partition: Partition) -> torch.Tensor:
        """Return token -> block-id mapping over the flattened transformer sequence."""
        token_block_map = torch.empty(partition.total_tokens, dtype=torch.long)
        for block in partition:
            indices = block.get_token_indices(partition.Ft, partition.Ht, partition.Wt)
            token_block_map[indices] = int(block.block_id)
        return token_block_map

    def describe_step_plan(self, step_plan: StepPlan, partition: Partition, step_idx: int) -> StepPlanDebugInfo:
        active_block_ids = set(step_plan.active_block_ids)
        active_block_count = len(active_block_ids)
        total_block_count = len(partition.blocks)
        active_token_count = int(step_plan.active_token_indices.numel())
        frozen_token_count = int(step_plan.frozen_token_indices.numel())
        return StepPlanDebugInfo(
            phase=step_plan.phase,
            step_idx=step_idx,
            active_block_count=active_block_count,
            frozen_block_count=max(0, total_block_count - active_block_count),
            active_token_count=active_token_count,
            frozen_token_count=frozen_token_count,
            total_token_count=partition.total_tokens,
            is_full_compute=step_plan.is_full_compute,
        )

    def _select_active_block_ids(
        self,
        partition: Partition,
        block_intervals: Dict[int, int],
        selective_step_idx: int,
    ) -> list[int]:
        active_block_ids: list[int] = []
        for block in partition:
            tau = int(block_intervals.get(block.block_id, 1))
            tau = max(1, tau)
            if selective_step_idx % tau == 0:
                active_block_ids.append(block.block_id)
        return active_block_ids

    @staticmethod
    def _concat_block_tokens(active_block_ids: list[int], token_indices_by_block: Dict[int, torch.Tensor]) -> torch.Tensor:
        tokens = []
        for block_id in active_block_ids:
            block_tokens = token_indices_by_block.get(block_id)
            if block_tokens is None or block_tokens.numel() == 0:
                continue
            tokens.append(block_tokens)
        if not tokens:
            return torch.empty(0, dtype=torch.long)
        return torch.unique(torch.cat(tokens, dim=0).sort()[0])


# Backward-compatible alias.
StepScheduler = SelectiveStepScheduler
