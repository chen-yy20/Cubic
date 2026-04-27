"""Straightness evaluation for Cubic anchor steps.

This module is evaluation-only. It does not participate in the acceleration
path. It estimates local curvature with a two-anchor finite difference of the
velocity field and compares it against endpoint prediction drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .visualizer import CubicVisualizer


@dataclass
class StraightnessRecord:
    step: int
    t: float
    dt: float
    drift_map: torch.Tensor
    curvature_gt_map: torch.Tensor
    proxy_curvature_map: torch.Tensor
    straightness_map: torch.Tensor
    ratio_map: torch.Tensor
    mean_drift: float
    mean_curvature_gt: float
    mean_proxy_curvature: float
    mean_ratio: float
    corr: float


class StraightnessEvaluator:
    """Evaluation-only monitor for verifying the straightness proxy relation."""

    def __init__(self, config):
        self.config = config
        self.visualizer = CubicVisualizer(config)
        self.prev_endpoint: Optional[torch.Tensor] = None
        self.prev_velocity: Optional[torch.Tensor] = None
        self.prev_t: Optional[float] = None
        self.records: list[StraightnessRecord] = []

    def update(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        t: float | torch.Tensor,
        step_idx: Optional[int] = None,
        velocity_fn: Optional[Callable[[torch.Tensor, float], torch.Tensor]] = None,
    ) -> Optional[StraightnessRecord]:
        """Record a new anchor sample.

        On the second and later calls, compute:
        - endpoint drift between consecutive anchors
        - independent finite-difference curvature from two velocities
        - drift-based curvature proxy
        - straightness score
        - ratio between drift proxy and independent curvature
        """
        t_cur = self._to_scalar_t(t)
        endpoint = x + (1.0 - t_cur) * v

        if self.prev_endpoint is None or self.prev_velocity is None or self.prev_t is None:
            self.prev_endpoint = endpoint.detach()
            self.prev_velocity = v.detach()
            self.prev_t = t_cur
            return None

        dt = abs(t_cur - float(self.prev_t))
        eps = 1e-8

        drift = endpoint - self.prev_endpoint
        drift_map = torch.linalg.vector_norm(drift, ord=2, dim=1).mean(dim=0)

        curvature_gt_map = self._compute_groundtruth_curvature(
            x=x,
            v=v,
            t_cur=t_cur,
            dt=dt,
            velocity_fn=velocity_fn,
        )

        velocity_norm_map = torch.linalg.vector_norm(v, ord=2, dim=1).mean(dim=0)
        proxy_denominator = (1.0 - t_cur) * dt + eps
        proxy_curvature_map = drift_map / proxy_denominator

        straightness_denominator = (1.0 - t_cur) * velocity_norm_map * dt + eps
        straightness_map = 1.0 - drift_map / straightness_denominator

        ratio_map = drift_map / (((1.0 - t_cur) * dt) * curvature_gt_map + eps)

        mean_drift = float(drift_map.mean().item())
        mean_curvature_gt = float(curvature_gt_map.mean().item())
        mean_proxy_curvature = float(proxy_curvature_map.mean().item())
        mean_ratio = float(ratio_map.mean().item())
        corr = self._pearson_corr(
            proxy_curvature_map.reshape(-1),
            curvature_gt_map.reshape(-1),
        )

        record = StraightnessRecord(
            step=int(step_idx) if step_idx is not None else -1,
            t=t_cur,
            dt=dt,
            drift_map=self._pool_to_token_space(drift_map),
            curvature_gt_map=self._pool_to_token_space(curvature_gt_map),
            proxy_curvature_map=self._pool_to_token_space(proxy_curvature_map),
            straightness_map=self._pool_to_token_space(straightness_map),
            ratio_map=self._pool_to_token_space(ratio_map),
            mean_drift=mean_drift,
            mean_curvature_gt=mean_curvature_gt,
            mean_proxy_curvature=mean_proxy_curvature,
            mean_ratio=mean_ratio,
            corr=corr,
        )
        self.records.append(record)

        self.prev_endpoint = endpoint.detach()
        self.prev_velocity = v.detach()
        self.prev_t = t_cur
        return record

    def reset(self):
        self.prev_endpoint = None
        self.prev_velocity = None
        self.prev_t = None
        self.records = []

    def export_visualization(self, output_dir: str):
        """Backward-compatible entry; visualization is centralized in cubic_core.visualizer."""
        self.visualizer.export_straightness_records(self.records, output_dir)

    def _compute_groundtruth_curvature(self,
                                       x: torch.Tensor,
                                       v: torch.Tensor,
                                       t_cur: float,
                                       dt: float,
                                       velocity_fn: Optional[Callable[[torch.Tensor, float], torch.Tensor]]):
        """Approximate kappa via definition (3.1): d_t v = partial_t v + (nabla_x v)·v."""
        eps = 1e-8
        if velocity_fn is None:
            velocity_delta = (v - self.prev_velocity) / (dt + eps)
            return torch.linalg.vector_norm(velocity_delta, ord=2, dim=1).mean(dim=0)

        time_eps = float(getattr(self.config, 'straightness_time_eps', 1e-3))
        space_eps = float(getattr(self.config, 'straightness_space_eps', 1e-3))
        time_eps = max(time_eps, 1e-6)
        space_eps = max(space_eps, 1e-6)

        t_plus = min(1.0, t_cur + time_eps)
        t_minus = max(0.0, t_cur - time_eps)
        eff_dt = max(t_plus - t_minus, 1e-8)

        v_t_plus = velocity_fn(x, t_plus)
        v_t_minus = velocity_fn(x, t_minus)
        partial_t = (v_t_plus - v_t_minus) / eff_dt

        v_norm = torch.linalg.vector_norm(v, ord=2, dim=1, keepdim=True)
        direction = v / (v_norm + eps)
        x_plus = x + space_eps * direction
        x_minus = x - space_eps * direction

        v_x_plus = velocity_fn(x_plus, t_cur)
        v_x_minus = velocity_fn(x_minus, t_cur)
        dir_derivative = (v_x_plus - v_x_minus) / (2.0 * space_eps)
        advective = dir_derivative * v_norm

        total_derivative = partial_t + advective
        return torch.linalg.vector_norm(total_derivative, ord=2, dim=1).mean(dim=0)

    def _pool_to_token_space(self, curvature_fhw: torch.Tensor) -> torch.Tensor:
        pt, ph, pw = getattr(self.config, 'patch_size', (1, 2, 2))
        pt, ph, pw = int(pt), int(ph), int(pw)
        if pt <= 0 or ph <= 0 or pw <= 0:
            raise ValueError('patch_size must be positive')

        f_dim, h_dim, w_dim = curvature_fhw.shape
        f_eff = (f_dim // pt) * pt
        h_eff = (h_dim // ph) * ph
        w_eff = (w_dim // pw) * pw
        if f_eff == 0 or h_eff == 0 or w_eff == 0:
            raise ValueError('curvature map is smaller than patch_size')

        x = curvature_fhw[:f_eff, :h_eff, :w_eff].float()
        x = x.unsqueeze(0).unsqueeze(0)
        pooled = F.avg_pool3d(
            x,
            kernel_size=(pt, ph, pw),
            stride=(pt, ph, pw),
        )
        return pooled.squeeze(0).squeeze(0)

    @staticmethod
    def _to_scalar_t(t: float | torch.Tensor) -> float:
        if isinstance(t, torch.Tensor):
            if t.numel() == 0:
                raise ValueError('empty tensor timestep')
            return float(t.detach().reshape(-1)[0].item())
        return float(t)

    @staticmethod
    def _pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.float().reshape(-1)
        b = b.float().reshape(-1)
        if a.numel() == 0 or b.numel() == 0:
            return 0.0
        a = a - a.mean()
        b = b - b.mean()
        denom = torch.sqrt((a * a).mean() * (b * b).mean()) + 1e-8
        return float(((a * b).mean() / denom).item())