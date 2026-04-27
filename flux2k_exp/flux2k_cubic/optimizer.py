from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from flexflow import Phase, StepPlan


@dataclass
class Flux2KPolicyPlan:
    # Per-block cooldown steps in selective stage.
    # For block b with cooldown c_b and selective length S:
    # - freeze first S-c_b steps
    # - active last c_b steps
    cooldown_by_block: Dict[int, int]
    selective_steps: int
    anchor_steps: int
    total_steps: int

    def as_interval_map_for_visualize(self) -> Dict[int, int]:
        # 仅用于复用现有可视化接口，不作为真实调度逻辑。
        # cooldown语义映射到“近似更新间隔”: cooldown越大，等效tau越小。
        out: Dict[int, int] = {}
        sel = max(1, int(self.selective_steps))
        for block_id, cooldown in self.cooldown_by_block.items():
            c = max(0, min(sel, int(cooldown)))
            if c == 0:
                out[block_id] = sel + 1
            else:
                out[block_id] = max(1, sel - c + 1)
        return out


class Flux2KAdaptiveOptimizer:
    """flux2k专用策略优化器。

    目标：
        - 不做通用周期tau，而是输出 block 级 cooldown 步数 c_b。
        - 对每个 block b：冻结 selective 前 S-c_b 步，激活最后 c_b 步。
        - 当 S=2 时自然对应 00/01/11。
    """

    def __init__(self, config):
        self.config = config
        self.max_cooldown_cap = int(getattr(config, "flux2k_max_cooldown_steps", 2))
        self.eps = 1e-8

    def optimize(
        self,
        partition,
        block_curvatures: Dict[int, float],
        total_steps: int,
        anchor_steps: int = 2,
    ) -> Flux2KPolicyPlan:
        selective_steps = max(0, int(total_steps) - int(anchor_steps))
        if selective_steps == 0:
            return Flux2KPolicyPlan(
                cooldown_by_block={block.block_id: 0 for block in partition},
                selective_steps=0,
                anchor_steps=int(anchor_steps),
                total_steps=int(total_steps),
            )

        max_cooldown = max(0, min(selective_steps, self.max_cooldown_cap))
        cooldown_by_block = self._assign_block_cooldown(
            partition=partition,
            block_curvatures=block_curvatures,
            max_cooldown=max_cooldown,
            selective_steps=selective_steps,
        )

        return Flux2KPolicyPlan(
            cooldown_by_block=cooldown_by_block,
            selective_steps=selective_steps,
            anchor_steps=int(anchor_steps),
            total_steps=int(total_steps),
        )

    def build_step_plan(self, partition, plan: Flux2KPolicyPlan, step_idx: int) -> StepPlan | None:
        # Anchor 阶段返回 None，由调用方走 full forward。
        if step_idx < plan.anchor_steps:
            return None

        selective_local_idx = step_idx - plan.anchor_steps
        if selective_local_idx < 0 or selective_local_idx >= plan.selective_steps:
            return None

        token_indices_by_block = partition.get_all_token_indices()
        active_block_ids = []
        token_chunks = []

        for block in partition:
            cooldown = int(plan.cooldown_by_block.get(block.block_id, plan.selective_steps))
            cooldown = max(0, min(plan.selective_steps, cooldown))
            if self._is_block_active(cooldown, selective_local_idx, plan.selective_steps):
                active_block_ids.append(block.block_id)
                block_tokens = token_indices_by_block.get(block.block_id)
                if block_tokens is not None and block_tokens.numel() > 0:
                    token_chunks.append(block_tokens)

        if token_chunks:
            active_token_indices = torch.unique(torch.cat(token_chunks, dim=0).sort()[0])
        else:
            active_token_indices = torch.empty(0, dtype=torch.long)

        all_tokens = torch.arange(partition.total_tokens, dtype=torch.long)
        if active_token_indices.numel() > 0:
            frozen_mask = torch.ones(partition.total_tokens, dtype=torch.bool)
            frozen_mask[active_token_indices] = False
            frozen_token_indices = all_tokens[frozen_mask]
        else:
            frozen_token_indices = all_tokens

        return StepPlan(
            phase=Phase.CRUISE_SELECTIVE,
            active_block_ids=active_block_ids,
            active_token_indices=active_token_indices,
            frozen_token_indices=frozen_token_indices,
            is_full_compute=False,
        )

    def _assign_block_cooldown(
        self,
        partition,
        block_curvatures: Dict[int, float],
        max_cooldown: int,
        selective_steps: int,
    ) -> Dict[int, int]:
        if selective_steps <= 0 or max_cooldown <= 0:
            return {block.block_id: 0 for block in partition}

        block_ids = [block.block_id for block in partition]
        curv = torch.tensor(
            [max(float(block_curvatures.get(block_id, 0.0)), 0.0) for block_id in block_ids],
            dtype=torch.float32,
        )
        if curv.numel() == 0:
            return {block_id: 0 for block_id in block_ids}

        vmax = float(curv.max().item())
        if vmax <= self.eps:
            return {block_id: 0 for block_id in block_ids}

        # Rank-based bucketization to maximize relative performance gain in short schedules.
        # This ensures low-curvature blocks can fall into cooldown=0 bucket (full freeze in selective stage).
        # For S=2 (4-step inference), it naturally realizes 00/01/11-style levels.
        n = len(block_ids)
        order = torch.argsort(curv, descending=False)
        cooldown = torch.zeros(n, dtype=torch.int64)
        for rank, idx in enumerate(order.tolist()):
            # rank_ratio in (0,1], then uniformly bucket to [0, max_cooldown]
            rank_ratio = float(rank + 1) / float(n)
            c = int(torch.floor(torch.tensor(rank_ratio * (max_cooldown + 1.0) - self.eps)).item())
            cooldown[idx] = max(0, min(max_cooldown, c))

        # Guarantee at least one cooldown=0 block when there are at least 2 blocks,
        # so the optimizer can actually exploit full selective-stage freeze.
        if n >= 2 and int((cooldown == 0).sum().item()) == 0:
            min_idx = int(torch.argmin(curv).item())
            cooldown[min_idx] = 0

        return {block_id: int(c) for block_id, c in zip(block_ids, cooldown.tolist())}

    @staticmethod
    def _is_block_active(cooldown: int, selective_local_idx: int, selective_steps: int) -> bool:
        # Active on the last `cooldown` selective steps.
        if cooldown <= 0:
            return False
        return selective_local_idx >= selective_steps - cooldown
