"""Block-wise update frequency optimizer for Cubic selective phase.

Implements implement.md section 2.5:
- Continuous solution: tau_b ∝ kappa_b^{-alpha/(beta+1)}
- Normalize with mean(1/tau_b) = 1/target_speedup
- Discretize with round + clip to [tau_min, tau_max]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class FrequencyPlan:
    """Discrete per-block update interval plan."""

    block_intervals: Dict[int, int]
    mean_interval: float
    mean_update_rate: float
    achieved_speedup: float
    min_interval: int
    max_interval: int


class UpdateFrequencyOptimizer:
    """Map block curvature to update interval tau_b."""

    def __init__(self, config):
        self.alpha = float(getattr(config, 'alpha', 2.0))
        self.beta = float(getattr(config, 'beta', 4.0))
        self.tau_min = int(getattr(config, 'tau_min', 1))
        self.tau_max = int(getattr(config, 'tau_max', 10))
        self.target_speedup = float(getattr(config, 'target_speedup', 2.0))
        self.eps = 1e-8

    def optimize(self, partition, block_curvatures: Dict[int, float]) -> FrequencyPlan:
        """Build per-block tau plan from curvature.

        Args:
            partition: Partition object with iterable blocks.
            block_curvatures: {block_id: kappa_b} mapping.

        Returns:
            FrequencyPlan containing integer intervals per block.
        """
        blocks = list(partition)
        if not blocks:
            return FrequencyPlan(
                block_intervals={},
                mean_interval=0.0,
                mean_update_rate=0.0,
                achieved_speedup=0.0,
                min_interval=self.tau_min,
                max_interval=self.tau_max,
            )

        kappas = []
        block_ids = []
        for block in blocks:
            block_ids.append(block.block_id)
            kappa = float(block_curvatures.get(block.block_id, 0.0))
            kappas.append(max(kappa, self.eps))

        kappa_arr = np.asarray(kappas, dtype=np.float64)
        exponent = -self.alpha / (self.beta + 1.0)
        raw_tau = np.power(kappa_arr, exponent)

        # Normalize to satisfy mean(1/tau) = 1/target_speedup.
        inv_raw_mean = float(np.mean(1.0 / np.maximum(raw_tau, self.eps)))
        target = max(self.target_speedup, 1.0)
        scale = target * inv_raw_mean
        tau_cont = scale * raw_tau

        tau_discrete = np.rint(tau_cont)
        tau_discrete = np.clip(tau_discrete, self.tau_min, self.tau_max).astype(np.int32)

        update_rates = 1.0 / np.maximum(tau_discrete.astype(np.float64), self.eps)
        mean_update_rate = float(np.mean(update_rates))
        achieved_speedup = float(1.0 / max(mean_update_rate, self.eps))

        block_intervals = {
            block_id: int(interval)
            for block_id, interval in zip(block_ids, tau_discrete.tolist())
        }

        return FrequencyPlan(
            block_intervals=block_intervals,
            mean_interval=float(np.mean(tau_discrete).item()),
            mean_update_rate=mean_update_rate,
            achieved_speedup=achieved_speedup,
            min_interval=int(np.min(tau_discrete).item()),
            max_interval=int(np.max(tau_discrete).item()),
        )
