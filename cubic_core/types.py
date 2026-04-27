"""Core data structures for Cubic system.

Following cubic_core/docs/implement.md section 4.2-4.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import torch


class Phase(Enum):
    """Inference phase classification."""
    WARMUP = "warmup"
    CRUISE_ANCHOR = "cruise_anchor"
    CRUISE_SELECTIVE = "cruise_selective"
    COOLDOWN = "cooldown"


@dataclass
class Block:
    """A spatial block in token space.
    
    Coordinates: (Ft, Ht, Wt) = token grid dimensions.
    Block defines a rectangular region in (f, h, w).
    """
    block_id: int
    f_start: int
    f_end: int
    h_start: int
    h_end: int
    w_start: int
    w_end: int
    level: int = 0

    @property
    def token_count(self) -> int:
        """Total number of tokens in this block."""
        return (self.f_end - self.f_start) * \
               (self.h_end - self.h_start) * \
               (self.w_end - self.w_start)

    @property
    def f_size(self) -> int:
        return self.f_end - self.f_start

    @property
    def h_size(self) -> int:
        return self.h_end - self.h_start

    @property
    def w_size(self) -> int:
        return self.w_end - self.w_start

    @property
    def sizes(self) -> tuple[int, int, int]:
        return self.f_size, self.h_size, self.w_size

    def get_token_indices(self, Ft: int, Ht: int, Wt: int) -> torch.Tensor:
        """Return all token indices in this block after flattening.
        
        Token flattening order: (f, h, w) → f * Ht * Wt + h * Wt + w.
        
        Args:
            Ft, Ht, Wt: Token grid dimensions.
        
        Returns:
            LongTensor of shape (token_count,).
        """
        f_idx = torch.arange(self.f_start, self.f_end, dtype=torch.long)
        h_idx = torch.arange(self.h_start, self.h_end, dtype=torch.long)
        w_idx = torch.arange(self.w_start, self.w_end, dtype=torch.long)
        
        # Vectorized grid generation
        ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing='ij')
        indices = ff * Ht * Wt + hh * Wt + ww
        return indices.reshape(-1)


@dataclass
class Partition:
    """Complete partitioning scheme: non-overlapping blocks covering all tokens."""
    
    blocks: List[Block]
    Ft: int  # Token grid time dimension
    Ht: int  # Token grid height
    Wt: int  # Token grid width
    level_counts: Optional[Dict[int, int]] = None
    selected_block_size: Optional[tuple[int, int, int]] = None

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks)

    def get_all_token_indices(self) -> Dict[int, torch.Tensor]:
        """Return {block_id: token_indices} mapping."""
        return {
            b.block_id: b.get_token_indices(self.Ft, self.Ht, self.Wt)
            for b in self.blocks
        }

    @property
    def total_tokens(self) -> int:
        return self.Ft * self.Ht * self.Wt

    def validate(self) -> None:
        """Verify partition covers all tokens with no overlaps."""
        # For spatial partitions, validate non-overlapping coverage
        total_coverage = 0
        for block in self.blocks:
            if block.f_start < 0 or block.h_start < 0 or block.w_start < 0:
                raise ValueError(f"Block {block.block_id} has negative start index")
            if block.f_end > self.Ft or block.h_end > self.Ht or block.w_end > self.Wt:
                raise ValueError(f"Block {block.block_id} exceeds grid bounds")
            if block.f_end <= block.f_start or block.h_end <= block.h_start or block.w_end <= block.w_start:
                raise ValueError(f"Block {block.block_id} has zero or negative size")
            total_coverage += block.token_count

        if total_coverage != self.total_tokens:
            raise ValueError(
                f"Partition incomplete: covered {total_coverage} tokens, "
                f"expected {self.total_tokens}"
            )

    def level_summary(self) -> Dict[int, int]:
        """Return block count per level."""
        if self.level_counts is not None:
            return dict(self.level_counts)

        summary: Dict[int, int] = {}
        for block in self.blocks:
            summary[block.level] = summary.get(block.level, 0) + 1
        return summary

    def size_statistics(self) -> Dict[str, float]:
        """Return basic block size statistics."""
        if not self.blocks:
            return {
                'count': 0,
                'min_tokens': 0.0,
                'max_tokens': 0.0,
                'mean_tokens': 0.0,
                'median_tokens': 0.0,
            }

        token_counts = torch.tensor([block.token_count for block in self.blocks], dtype=torch.float32)
        return {
            'count': float(len(self.blocks)),
            'min_tokens': float(token_counts.min().item()),
            'max_tokens': float(token_counts.max().item()),
            'mean_tokens': float(token_counts.mean().item()),
            'median_tokens': float(token_counts.median().item()),
        }


@dataclass
class StepPlan:
    """Single-step inference execution plan."""
    
    phase: Phase
    active_block_ids: List[int]
    active_token_indices: torch.Tensor
    frozen_token_indices: torch.Tensor
    is_full_compute: bool = False

    def __post_init__(self):
        # Ensure indices are on CPU and sorted
        if self.active_token_indices.device.type != 'cpu':
            self.active_token_indices = self.active_token_indices.cpu()
        if self.frozen_token_indices.device.type != 'cpu':
            self.frozen_token_indices = self.frozen_token_indices.cpu()
        
        if self.active_token_indices.numel() > 0:
            self.active_token_indices = torch.sort(self.active_token_indices)[0]
        if self.frozen_token_indices.numel() > 0:
            self.frozen_token_indices = torch.sort(self.frozen_token_indices)[0]
