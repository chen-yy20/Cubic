"""Adaptive token partitioning for Cubic.

Default behavior now prefers a morphology-guided non-uniform spatial split:
- estimate a foreground mask from the time-averaged curvature map
- clean the mask with opening / closing / dilation / erosion
- recursively split mixed regions so foreground and background separate better
- keep foreground blocks finer than background blocks when possible

The previous global-uniform block-size search is still available as a fallback
or by setting ``partition_mode='uniform'`` on the config object.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .types import Block, Partition

_DEFAULT_PRESET: Dict[str, int | float | str] = {
    'partition_mode': 'morphology',
    'uniform_common_factor_blocks': False,
    'min_block_size': 2,
    'max_block_size': 16,
    'cv_threshold': 0.5,
    'min_block_tokens': 1,
    'max_block_tokens': 1 << 30,
    'max_blocksize_candidates': 36,
    'video_spatial_max_tokens': 1 << 30,
    'foreground_threshold_quantile': 0.8,
    'foreground_mixed_ratio_low': 0.15,
    'foreground_mixed_ratio_high': 0.85,
    'foreground_focus_block_size': 4,
    'morph_open_iters': 1,
    'morph_close_iters': 1,
    'morph_dilate_iters': 1,
    'morph_erode_iters': 0,
    'foreground_min_component_area': 4,
}


@dataclass
class SpatialRegion:
    h_start: int
    h_end: int
    w_start: int
    w_end: int
    level: int

    @property
    def h_size(self) -> int:
        return self.h_end - self.h_start

    @property
    def w_size(self) -> int:
        return self.w_end - self.w_start

    @property
    def area(self) -> int:
        return self.h_size * self.w_size


@dataclass
class PartitionDebugInfo:
    """Debug summary for chosen partition."""

    block_count: int
    chosen_block_size: Tuple[int, int, int]
    block_token_min: int
    block_token_max: int
    block_token_mean: float
    mean_block_cv: float
    p90_block_cv: float
    level_counts: Dict[int, int]


class AdaptivePartitioner:
    """Partition curvature-guided token grids into non-overlapping blocks."""

    def __init__(self, config):
        self.partition_mode = str(getattr(config, 'partition_mode', _DEFAULT_PRESET['partition_mode'])).lower()
        self.uniform_common_factor_blocks = bool(
            getattr(config, 'uniform_common_factor_blocks', _DEFAULT_PRESET['uniform_common_factor_blocks'])
        )
        self.min_block_size = int(getattr(config, 'min_block_size', _DEFAULT_PRESET['min_block_size']))
        self.max_block_size = int(getattr(config, 'max_block_size', _DEFAULT_PRESET['max_block_size']))
        self.cv_threshold = float(getattr(config, 'cv_threshold', _DEFAULT_PRESET['cv_threshold']))
        self.min_block_tokens = int(getattr(config, 'min_block_tokens', _DEFAULT_PRESET['min_block_tokens']))
        self.max_block_tokens = int(getattr(config, 'max_block_tokens', _DEFAULT_PRESET['max_block_tokens']))
        self.video_spatial_max_tokens = int(
            getattr(config, 'video_spatial_max_tokens', _DEFAULT_PRESET['video_spatial_max_tokens'])
        )
        self.max_candidates = int(
            getattr(config, 'max_blocksize_candidates', _DEFAULT_PRESET['max_blocksize_candidates'])
        )
        self.foreground_threshold_quantile = float(
            getattr(config, 'foreground_threshold_quantile', _DEFAULT_PRESET['foreground_threshold_quantile'])
        )
        self.foreground_mixed_ratio_low = float(
            getattr(config, 'foreground_mixed_ratio_low', _DEFAULT_PRESET['foreground_mixed_ratio_low'])
        )
        self.foreground_mixed_ratio_high = float(
            getattr(config, 'foreground_mixed_ratio_high', _DEFAULT_PRESET['foreground_mixed_ratio_high'])
        )
        self.foreground_focus_block_size = int(
            getattr(config, 'foreground_focus_block_size', _DEFAULT_PRESET['foreground_focus_block_size'])
        )
        self.morph_open_iters = int(getattr(config, 'morph_open_iters', _DEFAULT_PRESET['morph_open_iters']))
        self.morph_close_iters = int(getattr(config, 'morph_close_iters', _DEFAULT_PRESET['morph_close_iters']))
        self.morph_dilate_iters = int(
            getattr(config, 'morph_dilate_iters', _DEFAULT_PRESET['morph_dilate_iters'])
        )
        self.morph_erode_iters = int(getattr(config, 'morph_erode_iters', _DEFAULT_PRESET['morph_erode_iters']))
        self.foreground_min_component_area = int(
            getattr(config, 'foreground_min_component_area', _DEFAULT_PRESET['foreground_min_component_area'])
        )
        self.eps = 1e-8

    def partition(
        self,
        Ft: int,
        Ht: int,
        Wt: int,
        curvature_map: Optional[torch.Tensor] = None,
    ) -> Partition:
        """Generate a partition for the token grid."""
        if curvature_map is not None and curvature_map.shape != (Ft, Ht, Wt):
            raise ValueError(
                f'Curvature map shape {tuple(curvature_map.shape)} does not match '
                f'token grid ({Ft}, {Ht}, {Wt})'
            )

        if curvature_map is None or self.partition_mode == 'uniform':
            partition = self._build_uniform_partition(Ft, Ht, Wt, curvature_map)
            partition.validate()
            return partition

        morph_partition = self._build_morphology_partition(Ft, Ht, Wt, curvature_map.detach().float())
        if morph_partition is not None:
            morph_partition.validate()
            return morph_partition

        partition = self._build_uniform_partition(Ft, Ht, Wt, curvature_map)
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

    def _build_uniform_partition(
        self,
        ft: int,
        ht: int,
        wt: int,
        curvature_map: Optional[torch.Tensor],
    ) -> Partition:
        if curvature_map is None:
            chosen = self._default_block_size(ft, ht, wt)
        elif self.uniform_common_factor_blocks:
            chosen = self._common_factor_block_size(ft, ht, wt)
        else:
            chosen = self._select_block_size(curvature_map.detach().float())

        blocks = self._tile_uniform(ft, ht, wt, chosen)
        return Partition(
            blocks=blocks,
            Ft=ft,
            Ht=ht,
            Wt=wt,
            level_counts={0: len(blocks)},
            selected_block_size=chosen,
            is_uniform=True,
            partition_mode='uniform',
        )

    def _build_morphology_partition(
        self,
        ft: int,
        ht: int,
        wt: int,
        curvature_map: torch.Tensor,
    ) -> Partition | None:
        spatial_map = curvature_map.mean(dim=0)
        if spatial_map.numel() == 0:
            return None

        fg_mask = self._build_foreground_mask(spatial_map)
        regions = self._recursive_partition_spatial(
            spatial_map=spatial_map,
            fg_mask=fg_mask,
            temporal_size=ft,
            region=SpatialRegion(h_start=0, h_end=ht, w_start=0, w_end=wt, level=0),
        )
        if not regions:
            return None

        blocks = [
            Block(
                block_id=idx,
                f_start=0,
                f_end=ft,
                h_start=region.h_start,
                h_end=region.h_end,
                w_start=region.w_start,
                w_end=region.w_end,
                level=region.level,
            )
            for idx, region in enumerate(regions)
        ]
        return Partition(
            blocks=blocks,
            Ft=ft,
            Ht=ht,
            Wt=wt,
            level_counts=self._summarize_levels(blocks),
            selected_block_size=self._representative_block_size(blocks),
            is_uniform=False,
            partition_mode='morphology',
        )

    def _build_foreground_mask(self, spatial_map: torch.Tensor) -> torch.Tensor:
        values = spatial_map.detach().float()
        finite_mask = torch.isfinite(values)
        if not bool(finite_mask.any()):
            return torch.zeros_like(values, dtype=torch.bool)

        valid = values[finite_mask]
        if valid.numel() == 0:
            return torch.zeros_like(values, dtype=torch.bool)

        v_min = float(valid.min().item())
        v_max = float(valid.max().item())
        if abs(v_max - v_min) <= self.eps:
            return torch.zeros_like(values, dtype=torch.bool)

        quantile = min(max(self.foreground_threshold_quantile, 0.0), 1.0)
        threshold = float(torch.quantile(valid, quantile).item())
        fg_mask = values >= threshold
        fg_mask = torch.where(finite_mask, fg_mask, torch.zeros_like(fg_mask))

        original_mask = fg_mask.clone()
        fg_mask = self._binary_open(fg_mask, self.morph_open_iters)
        fg_mask = self._binary_close(fg_mask, self.morph_close_iters)
        fg_mask = self._binary_dilate(fg_mask, self.morph_dilate_iters)
        fg_mask = self._binary_erode(fg_mask, self.morph_erode_iters)
        fg_mask = self._remove_small_components(fg_mask)

        if not bool(fg_mask.any()) and bool(original_mask.any()):
            fg_mask = self._remove_small_components(original_mask, keep_largest=True)
        return fg_mask

    def _recursive_partition_spatial(
        self,
        spatial_map: torch.Tensor,
        fg_mask: torch.Tensor,
        temporal_size: int,
        region: SpatialRegion,
    ) -> List[SpatialRegion]:
        if region.area <= 0:
            return []

        if not self._should_split_region(spatial_map, fg_mask, temporal_size, region):
            return [region]

        split_axis, split_index = self._choose_split_axis_and_index(fg_mask, region)
        if split_axis is None or split_index is None:
            return [region]

        if split_axis == 'h':
            left = SpatialRegion(region.h_start, split_index, region.w_start, region.w_end, region.level + 1)
            right = SpatialRegion(split_index, region.h_end, region.w_start, region.w_end, region.level + 1)
        else:
            left = SpatialRegion(region.h_start, region.h_end, region.w_start, split_index, region.level + 1)
            right = SpatialRegion(region.h_start, region.h_end, split_index, region.w_end, region.level + 1)

        if left.area <= 0 or right.area <= 0:
            return [region]

        out: List[SpatialRegion] = []
        out.extend(self._recursive_partition_spatial(spatial_map, fg_mask, temporal_size, left))
        out.extend(self._recursive_partition_spatial(spatial_map, fg_mask, temporal_size, right))
        return out

    def _should_split_region(
        self,
        spatial_map: torch.Tensor,
        fg_mask: torch.Tensor,
        temporal_size: int,
        region: SpatialRegion,
    ) -> bool:
        h_size = region.h_size
        w_size = region.w_size
        if h_size <= 0 or w_size <= 0:
            return False
        if h_size == 1 and w_size == 1:
            return False

        fg_ratio = self._region_foreground_ratio(fg_mask, region)
        mean_cv = self._region_cv_2d(spatial_map, region)
        max_h, max_w = self._region_target_block_size(fg_ratio)
        spatial_tokens = region.area
        token_count = temporal_size * spatial_tokens
        mixed = self.foreground_mixed_ratio_low < fg_ratio < self.foreground_mixed_ratio_high

        size_pressure = h_size > max_h or w_size > max_w
        token_pressure = token_count > self.max_block_tokens or spatial_tokens > self.video_spatial_max_tokens
        detail_pressure = mean_cv > self.cv_threshold and max(h_size, w_size) > self.min_block_size
        purity_pressure = mixed and max(h_size, w_size) > 1
        return size_pressure or token_pressure or detail_pressure or purity_pressure

    def _region_target_block_size(self, fg_ratio: float) -> Tuple[int, int]:
        fg_focus = max(1, min(self.foreground_focus_block_size, self.max_block_size))
        bg_focus = max(1, self.max_block_size)
        mid_focus = max(fg_focus, min(bg_focus, (fg_focus + bg_focus) // 2))

        if fg_ratio >= self.foreground_mixed_ratio_high:
            target = fg_focus
        elif fg_ratio <= self.foreground_mixed_ratio_low:
            target = bg_focus
        else:
            target = mid_focus
        return target, target

    def _choose_split_axis_and_index(
        self,
        fg_mask: torch.Tensor,
        region: SpatialRegion,
    ) -> Tuple[str | None, int | None]:
        best_axis = None
        best_index = None
        best_score = None
        base_impurity = self._foreground_impurity(self._region_foreground_ratio(fg_mask, region), region.area)

        if region.h_size > 1:
            for split_h in range(region.h_start + 1, region.h_end):
                top = SpatialRegion(region.h_start, split_h, region.w_start, region.w_end, region.level + 1)
                bottom = SpatialRegion(split_h, region.h_end, region.w_start, region.w_end, region.level + 1)
                score = self._split_score(fg_mask, top, bottom)
                if best_score is None or score < best_score:
                    best_axis = 'h'
                    best_index = split_h
                    best_score = score

        if region.w_size > 1:
            for split_w in range(region.w_start + 1, region.w_end):
                left = SpatialRegion(region.h_start, region.h_end, region.w_start, split_w, region.level + 1)
                right = SpatialRegion(region.h_start, region.h_end, split_w, region.w_end, region.level + 1)
                score = self._split_score(fg_mask, left, right)
                if best_score is None or score < best_score:
                    best_axis = 'w'
                    best_index = split_w
                    best_score = score

        if best_axis is not None and best_score is not None and best_score < base_impurity + 1e-6:
            return best_axis, best_index

        if region.h_size >= region.w_size and region.h_size > 1:
            return 'h', region.h_start + region.h_size // 2
        if region.w_size > 1:
            return 'w', region.w_start + region.w_size // 2
        return None, None

    def _split_score(self, fg_mask: torch.Tensor, left: SpatialRegion, right: SpatialRegion) -> float:
        left_ratio = self._region_foreground_ratio(fg_mask, left)
        right_ratio = self._region_foreground_ratio(fg_mask, right)
        impurity = self._foreground_impurity(left_ratio, left.area) + self._foreground_impurity(right_ratio, right.area)
        balance_penalty = 0.01 * abs(left.area - right.area)
        edge_penalty = 0.05 * (
            float(min(left.h_size, left.w_size) <= 1) + float(min(right.h_size, right.w_size) <= 1)
        )
        return impurity + balance_penalty + edge_penalty

    def _foreground_impurity(self, fg_ratio: float, area: int) -> float:
        return min(fg_ratio, 1.0 - fg_ratio) * float(area)

    def _region_foreground_ratio(self, fg_mask: torch.Tensor, region: SpatialRegion) -> float:
        region_mask = fg_mask[region.h_start:region.h_end, region.w_start:region.w_end]
        if region_mask.numel() == 0:
            return 0.0
        return float(region_mask.float().mean().item())

    def _region_cv_2d(self, spatial_map: torch.Tensor, region: SpatialRegion) -> float:
        values = spatial_map[region.h_start:region.h_end, region.w_start:region.w_end]
        if values.numel() <= 1:
            return 0.0
        mean_val = float(values.mean().item())
        std_val = float(values.std(unbiased=False).item())
        return std_val / (abs(mean_val) + self.eps)

    def _remove_small_components(self, mask: torch.Tensor, keep_largest: bool = False) -> torch.Tensor:
        components = self._connected_components(mask)
        if not components:
            return torch.zeros_like(mask, dtype=torch.bool)

        min_area = max(1, self.foreground_min_component_area)
        kept = [comp for comp in components if len(comp) >= min_area]
        if keep_largest and not kept:
            kept = [max(components, key=len)]

        out = torch.zeros_like(mask, dtype=torch.bool)
        for comp in kept:
            for h_idx, w_idx in comp:
                out[h_idx, w_idx] = True
        return out

    def _connected_components(self, mask: torch.Tensor) -> List[List[Tuple[int, int]]]:
        mask_cpu = mask.detach().cpu()
        ht, wt = mask_cpu.shape
        visited = torch.zeros((ht, wt), dtype=torch.bool)
        components: List[List[Tuple[int, int]]] = []

        for h_idx in range(ht):
            for w_idx in range(wt):
                if visited[h_idx, w_idx] or not bool(mask_cpu[h_idx, w_idx]):
                    continue
                queue = deque([(h_idx, w_idx)])
                visited[h_idx, w_idx] = True
                coords: List[Tuple[int, int]] = []
                while queue:
                    cur_h, cur_w = queue.popleft()
                    coords.append((cur_h, cur_w))
                    for nxt_h, nxt_w in (
                        (cur_h - 1, cur_w),
                        (cur_h + 1, cur_w),
                        (cur_h, cur_w - 1),
                        (cur_h, cur_w + 1),
                    ):
                        if nxt_h < 0 or nxt_h >= ht or nxt_w < 0 or nxt_w >= wt:
                            continue
                        if visited[nxt_h, nxt_w] or not bool(mask_cpu[nxt_h, nxt_w]):
                            continue
                        visited[nxt_h, nxt_w] = True
                        queue.append((nxt_h, nxt_w))
                components.append(coords)
        return components

    def _binary_dilate(self, mask: torch.Tensor, iterations: int) -> torch.Tensor:
        out = mask.bool()
        for _ in range(max(0, iterations)):
            pooled = F.max_pool2d(out.float().unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)
            out = pooled.squeeze(0).squeeze(0) > 0
        return out

    def _binary_erode(self, mask: torch.Tensor, iterations: int) -> torch.Tensor:
        out = mask.bool()
        for _ in range(max(0, iterations)):
            inv = 1.0 - out.float().unsqueeze(0).unsqueeze(0)
            pooled = F.max_pool2d(inv, kernel_size=3, stride=1, padding=1)
            out = pooled.squeeze(0).squeeze(0) <= 0
        return out

    def _binary_open(self, mask: torch.Tensor, iterations: int) -> torch.Tensor:
        if iterations <= 0:
            return mask.bool()
        return self._binary_dilate(self._binary_erode(mask, iterations), iterations)

    def _binary_close(self, mask: torch.Tensor, iterations: int) -> torch.Tensor:
        if iterations <= 0:
            return mask.bool()
        return self._binary_erode(self._binary_dilate(mask, iterations), iterations)

    def _representative_block_size(self, blocks: Sequence[Block]) -> Tuple[int, int, int]:
        if not blocks:
            return 1, 1, 1
        f_sizes = torch.tensor([block.f_size for block in blocks], dtype=torch.int64)
        h_sizes = torch.tensor([block.h_size for block in blocks], dtype=torch.int64)
        w_sizes = torch.tensor([block.w_size for block in blocks], dtype=torch.int64)
        return (
            int(torch.median(f_sizes).item()),
            int(torch.median(h_sizes).item()),
            int(torch.median(w_sizes).item()),
        )

    def _summarize_levels(self, blocks: Sequence[Block]) -> Dict[int, int]:
        level_counts: Dict[int, int] = {}
        for block in blocks:
            level_counts[block.level] = level_counts.get(block.level, 0) + 1
        return level_counts

    def _select_block_size(self, curvature_map: torch.Tensor) -> Tuple[int, int, int]:
        ft, ht, wt = curvature_map.shape
        spatial_map = curvature_map.mean(dim=0)
        bh, bw = self._select_spatial_block_size(spatial_map, temporal_size=ft)
        return ft, bh, bw

    def _select_spatial_block_size(
        self,
        spatial_map: torch.Tensor,
        temporal_size: int,
    ) -> Tuple[int, int]:
        ht, wt = spatial_map.shape
        h_vals = self._axis_candidates(ht)
        w_vals = self._axis_candidates(wt)
        candidates = self._build_spatial_candidates(h_vals, w_vals)
        if not candidates:
            return self._default_block_size(temporal_size, ht, wt)[1:]

        metrics = []
        filtered_metrics = []
        for bh, bw in candidates:
            mean_cv, p90_cv, block_count = self._cv_stats_2d_uniform(spatial_map, (bh, bw))
            if mean_cv is None:
                continue
            metric = ((bh, bw), mean_cv, p90_cv, block_count)
            metrics.append(metric)

            token_count = int(temporal_size * bh * bw)
            spatial_tokens = int(bh * bw)
            if (
                self.min_block_tokens <= token_count <= self.max_block_tokens
                and spatial_tokens <= self.video_spatial_max_tokens
            ):
                filtered_metrics.append(metric)

        if not metrics:
            return self._default_block_size(temporal_size, ht, wt)[1:]

        metrics = filtered_metrics if filtered_metrics else metrics
        feasible = [m for m in metrics if m[2] <= self.cv_threshold]
        if feasible:
            feasible.sort(
                key=lambda m: (
                    m[0][0] * m[0][1],
                    -m[1],
                    -m[3],
                ),
                reverse=True,
            )
            return feasible[0][0]

        metrics.sort(
            key=lambda m: (
                m[2],
                m[1],
                -(m[0][0] * m[0][1]),
                m[3],
            )
        )
        return metrics[0][0]

    def _build_spatial_candidates(
        self,
        h_vals: Sequence[int],
        w_vals: Sequence[int],
    ) -> List[Tuple[int, int]]:
        candidates = [(h, w) for h in h_vals for w in w_vals]
        if len(candidates) > self.max_candidates:
            candidates.sort(key=lambda x: x[0] * x[1], reverse=True)
            step = max(1, len(candidates) // self.max_candidates)
            candidates = candidates[::step][:self.max_candidates]

        dedup: List[Tuple[int, int]] = []
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

        mid = max(1, min(dim, int(round((self.min_block_size + min(dim, self.max_block_size)) / 2))))
        vals.add(mid)
        return sorted({max(1, min(dim, v)) for v in vals})

    def _default_block_size(self, ft: int, ht: int, wt: int) -> Tuple[int, int, int]:
        f = max(1, min(ft, self.max_block_size))
        h = max(1, min(ht, self.max_block_size))
        w = max(1, min(wt, self.max_block_size))
        return f, h, w

    def _common_factor_block_size(self, ft: int, ht: int, wt: int) -> Tuple[int, int, int]:
        """Choose the largest cheap uniform block that exactly tiles each axis."""
        def largest_factor(dim: int) -> int:
            dim = max(1, int(dim))
            upper = max(1, min(dim, int(self.max_block_size)))
            lower = max(1, min(upper, int(self.min_block_size)))
            for value in range(upper, 0, -1):
                if dim % value == 0 and value >= lower:
                    return value
            for value in range(upper, 0, -1):
                if dim % value == 0:
                    return value
            return 1

        f = max(1, min(ft, self.max_block_size))
        h = largest_factor(ht)
        w = largest_factor(wt)
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

    def _cv_stats_2d_uniform(
        self,
        spatial_map: torch.Tensor,
        block_size: Tuple[int, int],
    ) -> Tuple[Optional[float], Optional[float], int]:
        block_values, block_mask = self._uniform_block_values_2d(spatial_map, block_size)
        if block_values.numel() == 0:
            return None, None, 0
        cv_tensor = self._cv_tensor_from_values(block_values, block_mask)
        if cv_tensor.numel() == 0:
            return None, None, 0
        return (
            float(cv_tensor.mean().item()),
            float(torch.quantile(cv_tensor, 0.9).item()),
            int(cv_tensor.numel()),
        )

    def _cv_tensor_from_values(self, block_values: torch.Tensor, block_mask: torch.Tensor) -> torch.Tensor:
        if block_values.numel() == 0:
            return torch.empty(0, dtype=block_values.dtype, device=block_values.device)

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
        block_tensor, block_mask = self._uniform_block_tensor(curvature_map, block_size)
        if block_tensor.numel() == 0:
            empty = torch.empty(0, 0, dtype=curvature_map.dtype, device=curvature_map.device)
            return empty, empty.bool()
        return block_tensor.reshape(block_tensor.shape[0], -1), block_mask.reshape(block_mask.shape[0], -1)

    def _uniform_block_tensor(
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
        values = values.permute(0, 2, 4, 1, 3, 5).reshape(nf * nh * nw, bf, bh, bw)

        valid = mask.view(nf, bf, nh, bh, nw, bw)
        valid = valid.permute(0, 2, 4, 1, 3, 5).reshape(nf * nh * nw, bf, bh, bw)
        return values, valid

    def _uniform_block_values_2d(
        self,
        spatial_map: torch.Tensor,
        block_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ht, wt = spatial_map.shape
        bh, bw = block_size
        nh = (ht + bh - 1) // bh
        nw = (wt + bw - 1) // bw
        if nh <= 0 or nw <= 0:
            empty = torch.empty(0, 0, dtype=spatial_map.dtype, device=spatial_map.device)
            return empty, empty.bool()

        ph = nh * bh
        pw = nw * bw
        padded = torch.zeros((ph, pw), dtype=spatial_map.dtype, device=spatial_map.device)
        padded[:ht, :wt] = spatial_map

        mask = torch.zeros((ph, pw), dtype=torch.bool, device=spatial_map.device)
        mask[:ht, :wt] = True

        values = padded.view(nh, bh, nw, bw).permute(0, 2, 1, 3).reshape(nh * nw, bh * bw)
        valid = mask.view(nh, bh, nw, bw).permute(0, 2, 1, 3).reshape(nh * nw, bh * bw)
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
        if any(block.f_size != bf or block.h_size not in (bh, min(bh, shape[1])) or block.w_size not in (bw, min(bw, shape[2])) for block in blocks):
            return None

        ft, ht, wt = shape
        expected = ((ft + bf - 1) // bf) * ((ht + bh - 1) // bh) * ((wt + bw - 1) // bw)
        if expected != len(blocks):
            return None
        return bf, bh, bw

    def _infer_block_size_from_partition(self, partition: Partition) -> Tuple[int, int, int]:
        if not partition.blocks:
            return 1, 1, 1
        return self._representative_block_size(partition.blocks)


# Backward-compatible alias for older imports.
CurvatureGradedPartitioner = AdaptivePartitioner
