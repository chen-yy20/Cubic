"""Block-wise update frequency optimizer for Cubic selective phase.

This implementation follows the theoretical continuous solution:
    tau_b ∝ kappa_b^(-alpha / (beta + 1))

We solve the scale factor under a token-weighted update-rate constraint, then
apply a small discrete correction so integer taus stay close to the target
speedup after rounding.
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
        self.tau_min = 1
        self.tau_max = min(int(getattr(config, 'tau_max', 10)), int(getattr(config, 'anchor_interval', 10)))

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
                min_interval=self.tau_min,
                max_interval=self.tau_max,
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
        tau_discrete = self._discretize_tau(tau_cont, kappa_arr, token_count_arr)

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
        max_rate = 1.0 / max(float(self.tau_min), self.eps)
        min_rate = 1.0 / max(float(self.tau_max), self.eps)
        target_speedup = max(float(self.target_speedup), 1.0)
        return min(max(1.0 / target_speedup, min_rate), max_rate)

    def _solve_continuous_tau(self, kappa_arr: np.ndarray, token_counts: np.ndarray) -> np.ndarray:
        if kappa_arr.size == 0:
            return kappa_arr

        gamma = max(float(self.curvature_contrast_gamma), self.eps)
        exponent = self.alpha / (self.beta + 1.0)
        scaled_kappa = np.power(np.maximum(kappa_arr, self.eps), gamma)
        raw_tau = np.power(scaled_kappa, -exponent)
        target_mean_rate = self._target_mean_rate()
        safe_token_counts = np.maximum(token_counts.astype(np.float64), 1.0)
        total_tokens = float(np.sum(safe_token_counts))

        def _mean_rate(scale: float) -> float:
            tau = np.clip(scale * raw_tau, self.tau_min, self.tau_max)
            return float(np.sum(safe_token_counts / np.maximum(tau, self.eps)) / max(total_tokens, self.eps))

        low = 0.0
        high = 1.0
        while _mean_rate(high) > target_mean_rate + self.eps:
            high *= 2.0
            if high > 1e12:
                break

        for _ in range(80):
            mid = 0.5 * (low + high)
            if _mean_rate(mid) > target_mean_rate:
                low = mid
            else:
                high = mid

        return np.clip(high * raw_tau, self.tau_min, self.tau_max)

    def _discretize_tau(
        self,
        tau_cont: np.ndarray,
        kappa_arr: np.ndarray,
        token_counts: np.ndarray,
    ) -> np.ndarray:
        tau = np.rint(tau_cont)
        tau = np.clip(tau, self.tau_min, self.tau_max).astype(np.int32)
        if tau.size == 0:
            return tau

        safe_token_counts = np.maximum(token_counts.astype(np.float64), 1.0)
        total_tokens = float(np.sum(safe_token_counts))
        target_mean_rate = self._target_mean_rate()
        loss_weight = np.power(np.maximum(kappa_arr, self.eps), self.alpha)

        def _weighted_rate(current_tau: np.ndarray) -> float:
            return float(
                np.sum(safe_token_counts / np.maximum(current_tau.astype(np.float64), self.eps))
                / max(total_tokens, self.eps)
            )

        current_rate = _weighted_rate(tau)
        max_steps = int(np.sum(self.tau_max - tau) + np.sum(tau - self.tau_min)) + int(tau.size) + 1

        for _ in range(max_steps):
            current_gap = abs(current_rate - target_mean_rate)
            if current_gap <= 1e-12:
                break

            if current_rate > target_mean_rate:
                movable = np.flatnonzero(tau < self.tau_max)
                if movable.size == 0:
                    break

                next_tau = tau[movable].astype(np.float64) + 1.0
                rate_delta = safe_token_counts[movable] * (
                    1.0 / tau[movable].astype(np.float64) - 1.0 / next_tau
                ) / max(total_tokens, self.eps)
                loss_delta = loss_weight[movable] * (
                    np.power(next_tau, self.beta) - np.power(tau[movable].astype(np.float64), self.beta)
                )
                candidate_rate = current_rate - rate_delta
                improved = np.abs(candidate_rate - target_mean_rate) < current_gap - 1e-12
                if not np.any(improved):
                    break
                candidate_idx = movable[improved]
                candidate_rate = candidate_rate[improved]
                candidate_score = loss_delta[improved] / np.maximum(rate_delta[improved], self.eps)
                best_local = int(np.argmin(candidate_score))
                best_idx = int(candidate_idx[best_local])
                tau[best_idx] += 1
                current_rate = float(candidate_rate[best_local])
                continue

            movable = np.flatnonzero(tau > self.tau_min)
            if movable.size == 0:
                break

            next_tau = tau[movable].astype(np.float64) - 1.0
            rate_delta = safe_token_counts[movable] * (
                1.0 / next_tau - 1.0 / tau[movable].astype(np.float64)
            ) / max(total_tokens, self.eps)
            loss_drop = loss_weight[movable] * (
                np.power(tau[movable].astype(np.float64), self.beta) - np.power(next_tau, self.beta)
            )
            candidate_rate = current_rate + rate_delta
            improved = np.abs(candidate_rate - target_mean_rate) < current_gap - 1e-12
            if not np.any(improved):
                break
            candidate_idx = movable[improved]
            candidate_rate = candidate_rate[improved]
            candidate_score = loss_drop[improved] / np.maximum(rate_delta[improved], self.eps)
            best_local = int(np.argmax(candidate_score))
            best_idx = int(candidate_idx[best_local])
            tau[best_idx] -= 1
            current_rate = float(candidate_rate[best_local])

        return tau
