"""Block-wise update frequency optimizer for Cubic selective phase.

This implementation follows the theoretical continuous solution:
    tau_b proportional to kappa_b^(-alpha / (beta + 1))

It solves the scale factor under a token-weighted update-rate constraint, then
rounds and clamps intervals to legal positive integers.
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
    block_mean_interval: float = 0.0
    block_mean_update_rate: float = 0.0
    block_achieved_speedup: float = 0.0


class UpdateFrequencyOptimizer:
    """Map block curvature to update interval tau_b."""

    def __init__(self, config):
        self.alpha = float(getattr(config, 'alpha', 1.0))
        self.beta = float(getattr(config, 'beta', 2.0))
        self.target_speedup = float(getattr(config, 'target_speedup', 2.0))
        # Optional contrast gain: >1 amplifies curvature differences before tau mapping.
        self.curvature_contrast_gamma = float(getattr(config, 'curvature_contrast_gamma', 1.0))
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
                min_interval=1,
                max_interval=1,
                block_mean_interval=0.0,
                block_mean_update_rate=0.0,
                block_achieved_speedup=0.0,
            )

        kappas = []
        block_ids = []
        token_counts = []
        for block in blocks:
            block_ids.append(block.block_id)
            kappa = float(block_curvatures.get(block.block_id, 0.0))
            kappas.append(max(kappa, self.eps))
            token_counts.append(max(int(block.token_count), 1))

        kappa_arr = np.asarray(kappas, dtype=np.float64)
        token_count_arr = np.asarray(token_counts, dtype=np.float64)
        tau_cont = self._solve_continuous_tau(kappa_arr, token_count_arr)
        tau_discrete = self._discretize_tau(tau_cont)

        update_rates = 1.0 / np.maximum(tau_discrete.astype(np.float64), self.eps)
        total_tokens = float(np.sum(token_count_arr))
        weighted_interval = float(np.sum(token_count_arr * tau_discrete.astype(np.float64)) / max(total_tokens, self.eps))
        mean_update_rate = float(np.sum(token_count_arr * update_rates) / max(total_tokens, self.eps))
        achieved_speedup = float(1.0 / max(mean_update_rate, self.eps))
        block_mean_interval = float(np.mean(tau_discrete))
        block_mean_update_rate = float(np.mean(update_rates))
        block_achieved_speedup = float(1.0 / max(block_mean_update_rate, self.eps))

        block_intervals = {
            block_id: int(interval)
            for block_id, interval in zip(block_ids, tau_discrete.tolist())
        }

        return FrequencyPlan(
            block_intervals=block_intervals,
            mean_interval=weighted_interval,
            mean_update_rate=mean_update_rate,
            achieved_speedup=achieved_speedup,
            min_interval=int(np.min(tau_discrete).item()),
            max_interval=int(np.max(tau_discrete).item()),
            block_mean_interval=block_mean_interval,
            block_mean_update_rate=block_mean_update_rate,
            block_achieved_speedup=block_achieved_speedup,
        )

    def _target_mean_rate(self) -> float:
        target_speedup = max(float(self.target_speedup), 1.0)
        return 1.0 / target_speedup

    def _solve_continuous_tau(self, kappa_arr: np.ndarray, token_counts: np.ndarray) -> np.ndarray:
        if kappa_arr.size == 0:
            return kappa_arr

        gamma = max(float(self.curvature_contrast_gamma), self.eps)
        exponent = self.alpha / (self.beta + 1.0)
        scaled_kappa = np.power(np.maximum(kappa_arr, self.eps), gamma) # 曲率对比度增强，默认无增强为1，放大曲率差异
        raw_tau = np.power(scaled_kappa, -exponent)
        target_mean_rate = self._target_mean_rate()
        safe_token_counts = np.maximum(token_counts.astype(np.float64), 1.0)
        total_tokens = float(np.sum(safe_token_counts))
        inverse_raw_rate = float(
            np.sum(safe_token_counts / np.maximum(raw_tau, self.eps))
            / max(total_tokens, self.eps)
        )
        scale = inverse_raw_rate / max(target_mean_rate, self.eps)
        return scale * raw_tau

    @staticmethod
    def _discretize_tau(tau_cont: np.ndarray) -> np.ndarray:
        tau = np.rint(tau_cont)
        return np.maximum(tau, 1).astype(np.int32)
