"""Uniform 3D partitioning with adaptive block-size search.

Design goal (per user requirement):
- Cubic decides only one global block size (f, h, w)
- Then uniformly tiles the whole latent/token grid with this block size
- Selection criterion: each block should have internally consistent curvature
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .types import Block, Partition


@dataclass
class PartitionDebugInfo:
    """Debug summary for chosen uniform partition."""

    block_count: int
    chosen_block_size: Tuple[int, int, int]
    block_token_min: int
    block_token_max: int
    block_token_mean: float
    mean_block_cv: float
    p90_block_cv: float
    level_counts: Dict[int, int]


class AdaptivePartitioner:
    """Choose a global block size and apply uniform 3D tiling."""

    def __init__(self, config):
        self.min_block_size = int(getattr(config, 'min_block_size', 2))
        self.max_block_size = int(getattr(config, 'max_block_size', 16))
        self.cv_threshold = float(getattr(config, 'cv_threshold', 0.5))
        self.min_block_tokens = int(getattr(config, 'min_block_tokens', 1))
        self.max_block_tokens = int(getattr(config, 'max_block_tokens', 1 << 30))
        self.eps = 1e-8

        # Keep candidate search controllable.
        self.max_candidates = int(getattr(config, 'max_blocksize_candidates', 36))

    def partition(
        self,
        Ft: int,
        Ht: int,
        Wt: int,
        curvature_map: Optional[torch.Tensor] = None,
    ) -> Partition:
        """Generate partition by selecting one global block size."""
        if curvature_map is not None and curvature_map.shape != (Ft, Ht, Wt):
            raise ValueError(
                f'Curvature map shape {tuple(curvature_map.shape)} does not match '
                f'token grid ({Ft}, {Ht}, {Wt})'
            )

        if curvature_map is None:
            chosen = self._default_block_size(Ft, Ht, Wt)
        else:
            chosen = self._select_block_size(curvature_map.detach().float())

        blocks = self._tile_uniform(Ft, Ht, Wt, chosen)
        partition = Partition(
            blocks=blocks,
            Ft=Ft,
            Ht=Ht,
            Wt=Wt,
            level_counts={0: len(blocks)},
            selected_block_size=chosen,
        )
        partition.validate()
        return partition

    def describe_partition(
        self,
        partition: Partition,
        curvature_map: Optional[torch.Tensor] = None,
    ) -> PartitionDebugInfo:
        """Return stats for logging and debugging."""
        blocks = list(partition)
        if not blocks:
            chosen = partition.selected_block_size or (1, 1, 1)
            return PartitionDebugInfo(
                block_count=0,
                chosen_block_size=chosen,
                block_token_min=0,
                block_token_max=0,
                block_token_mean=0.0,
                mean_block_cv=0.0,
                p90_block_cv=0.0,
                level_counts={0: 0},
            )

        token_counts = torch.tensor([b.token_count for b in blocks], dtype=torch.float32)
        mean_cv = 0.0
        p90_cv = 0.0
        if curvature_map is not None:
            if curvature_map.shape != (partition.Ft, partition.Ht, partition.Wt):
                raise ValueError('curvature_map shape mismatch in describe_partition')
            cv_list = self._block_cv_list(curvature_map.detach().float(), blocks)
            if cv_list:
                cv_tensor = torch.tensor(cv_list, dtype=torch.float32)
                mean_cv = float(cv_tensor.mean().item())
                p90_cv = float(torch.quantile(cv_tensor, 0.9).item())

        chosen = partition.selected_block_size or self._infer_block_size_from_partition(partition)
        return PartitionDebugInfo(
            block_count=len(blocks),
            chosen_block_size=chosen,
            block_token_min=int(token_counts.min().item()),
            block_token_max=int(token_counts.max().item()),
            block_token_mean=float(token_counts.mean().item()),
            mean_block_cv=mean_cv,
            p90_block_cv=p90_cv,
            level_counts=partition.level_summary(),
        )

    def _select_block_size(self, curvature_map: torch.Tensor) -> Tuple[int, int, int]:
        ft, ht, wt = curvature_map.shape
        candidates = self._build_candidates(ft, ht, wt)
        if not candidates:
            return self._default_block_size(ft, ht, wt)

        metrics = []
        filtered_metrics = []
        for block_size in candidates:
            cv_tensor = self._block_cv_tensor_uniform(curvature_map, block_size)
            if cv_tensor.numel() == 0:
                continue
            mean_cv = float(cv_tensor.mean().item())
            p90_cv = float(torch.quantile(cv_tensor, 0.9).item())
            metric = (block_size, mean_cv, p90_cv, int(cv_tensor.numel()))
            metrics.append(metric)

            token_count = int(block_size[0] * block_size[1] * block_size[2])
            if self.min_block_tokens <= token_count <= self.max_block_tokens:
                filtered_metrics.append(metric)

        if not metrics:
            return self._default_block_size(ft, ht, wt)

        metrics = filtered_metrics if filtered_metrics else metrics

        # First, prefer candidates where most blocks are internally consistent.
        feasible = [m for m in metrics if m[2] <= self.cv_threshold]
        if feasible:
            feasible.sort(
                key=lambda m: (
                    m[0][0] * m[0][1] * m[0][2],
                    -m[1],
                    -m[3],
                ),
                reverse=True,
            )
            return feasible[0][0]

        # Otherwise choose the best consistency, then largest block volume.
        metrics.sort(
            key=lambda m: (
                m[2],  # lower p90_cv better
                m[1],  # lower mean_cv better
                -(m[0][0] * m[0][1] * m[0][2]),  # larger block better as tie-break
                m[3],  # fewer blocks better
            )
        )
        return metrics[0][0]

    def _build_candidates(self, ft: int, ht: int, wt: int) -> List[Tuple[int, int, int]]:
        f_vals = self._axis_candidates(ft)
        h_vals = self._axis_candidates(ht)
        w_vals = self._axis_candidates(wt)

        candidates: List[Tuple[int, int, int]] = []
        for f in f_vals:
            for h in h_vals:
                for w in w_vals:
                    candidates.append((f, h, w))

        # Keep runtime bounded.
        if len(candidates) > self.max_candidates:
            # Prioritize larger blocks first, then subsample.
            candidates.sort(key=lambda x: x[0] * x[1] * x[2], reverse=True)
            step = max(1, len(candidates) // self.max_candidates)
            candidates = candidates[::step][:self.max_candidates]

        # Remove duplicates while preserving order.
        dedup: List[Tuple[int, int, int]] = []
        seen = set()
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            dedup.append(item)
        return dedup

    def _axis_candidates(self, dim: int) -> List[int]:
        dim = int(dim)
        if dim <= 1:
            return [1]

        vals = {1, min(dim, self.max_block_size)}
        p = max(1, self.min_block_size)
        while p <= min(dim, self.max_block_size):
            vals.add(p)
            p *= 2

        # Include a near-even divisor around the middle to avoid only powers of 2.
        mid = max(1, min(dim, int(round((self.min_block_size + min(dim, self.max_block_size)) / 2))))
        vals.add(mid)

        # Clamp and sort
        out = sorted({max(1, min(dim, v)) for v in vals})
        return out

    def _default_block_size(self, ft: int, ht: int, wt: int) -> Tuple[int, int, int]:
        f = max(1, min(ft, self.max_block_size))
        h = max(1, min(ht, self.max_block_size))
        w = max(1, min(wt, self.max_block_size))
        return f, h, w

    def _tile_uniform(self, ft: int, ht: int, wt: int, block_size: Tuple[int, int, int]) -> List[Block]:
        bf, bh, bw = block_size
        blocks: List[Block] = []
        block_id = 0
        for f_start in range(0, ft, bf):
            f_end = min(f_start + bf, ft)
            for h_start in range(0, ht, bh):
                h_end = min(h_start + bh, ht)
                for w_start in range(0, wt, bw):
                    w_end = min(w_start + bw, wt)
                    blocks.append(Block(
                        block_id=block_id,
                        f_start=f_start,
                        f_end=f_end,
                        h_start=h_start,
                        h_end=h_end,
                        w_start=w_start,
                        w_end=w_end,
                        level=0,
                    ))
                    block_id += 1
        return blocks

    def _block_cv_list(self, curvature_map: torch.Tensor, blocks: Sequence[Block]) -> List[float]:
        # For uniform partitions generated by this class, use vectorized path
        # to avoid per-block kernel launches / host sync.
        uniform_size = self._uniform_block_size_if_applicable(curvature_map.shape, blocks)
        if uniform_size is not None:
            cv_tensor = self._block_cv_tensor_uniform(curvature_map, uniform_size)
            return [float(v) for v in cv_tensor.detach().cpu().tolist()]

        values: List[float] = []
        for block in blocks:
            region = curvature_map[
                block.f_start:block.f_end,
                block.h_start:block.h_end,
                block.w_start:block.w_end,
            ]
            if region.numel() <= 1:
                values.append(0.0)
                continue
            mean_val = float(region.mean().item())
            std_val = float(region.std(unbiased=False).item())
            cv = std_val / (abs(mean_val) + self.eps)
            values.append(cv)
        return values

    def _block_cv_tensor_uniform(
        self,
        curvature_map: torch.Tensor,
        block_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        block_values, block_mask = self._uniform_block_values(curvature_map, block_size)
        if block_values.numel() == 0:
            return torch.empty(0, dtype=curvature_map.dtype, device=curvature_map.device)

        valid = block_mask.to(dtype=block_values.dtype)
        count = valid.sum(dim=1)
        safe_count = count.clamp_min(1.0)
        sums = (block_values * valid).sum(dim=1)
        means = sums / safe_count
        sums_sq = (block_values * block_values * valid).sum(dim=1)
        variances = (sums_sq / safe_count - means * means).clamp_min(0.0)
        stds = torch.sqrt(variances)
        cvs = stds / (means.abs() + self.eps)
        return torch.where(count <= 1.0, torch.zeros_like(cvs), cvs)

    def _uniform_block_values(
        self,
        curvature_map: torch.Tensor,
        block_size: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ft, ht, wt = curvature_map.shape
        bf, bh, bw = block_size
        nf = (ft + bf - 1) // bf
        nh = (ht + bh - 1) // bh
        nw = (wt + bw - 1) // bw
        if nf <= 0 or nh <= 0 or nw <= 0:
            empty = torch.empty(0, 0, dtype=curvature_map.dtype, device=curvature_map.device)
            return empty, empty.bool()

        pf = nf * bf
        ph = nh * bh
        pw = nw * bw
        device = curvature_map.device
        dtype = curvature_map.dtype

        padded = torch.zeros((pf, ph, pw), dtype=dtype, device=device)
        padded[:ft, :ht, :wt] = curvature_map

        mask = torch.zeros((pf, ph, pw), dtype=torch.bool, device=device)
        mask[:ft, :ht, :wt] = True

        values = padded.view(nf, bf, nh, bh, nw, bw)
        values = values.permute(0, 2, 4, 1, 3, 5).reshape(nf * nh * nw, bf * bh * bw)

        valid = mask.view(nf, bf, nh, bh, nw, bw)
        valid = valid.permute(0, 2, 4, 1, 3, 5).reshape(nf * nh * nw, bf * bh * bw)
        return values, valid

    def _uniform_block_size_if_applicable(
        self,
        shape: Tuple[int, int, int],
        blocks: Sequence[Block],
    ) -> Optional[Tuple[int, int, int]]:
        if not blocks:
            return None

        bf = blocks[0].f_size
        bh = blocks[0].h_size
        bw = blocks[0].w_size
        ft, ht, wt = shape
        expected = ((ft + bf - 1) // bf) * ((ht + bh - 1) // bh) * ((wt + bw - 1) // bw)
        if expected != len(blocks):
            return None
        return bf, bh, bw

    def _infer_block_size_from_partition(self, partition: Partition) -> Tuple[int, int, int]:
        if not partition.blocks:
            return 1, 1, 1
        b = partition.blocks[0]
        return b.f_size, b.h_size, b.w_size


# Backward-compatible alias for older imports.
CurvatureGradedPartitioner = AdaptivePartitioner
