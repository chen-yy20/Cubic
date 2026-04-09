"""Straightness evaluation for Cubic anchor steps.

This module is evaluation-only. It does not participate in the acceleration
path. It estimates local curvature with a two-anchor finite difference of the
velocity field and compares it against endpoint prediction drift.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
        """Export overview PNGs for straightness verification."""
        if not self.records:
            return

        vis_dir = os.path.join(output_dir, 'straightness_vis')
        os.makedirs(vis_dir, exist_ok=True)
        self._clear_directory(vis_dir)

        self._save_metric_overview(os.path.join(vis_dir, 'straightness_overview.png'))
        self._save_heatmap_overview(os.path.join(vis_dir, 'straightness_heatmap_overview.png'))
        self._save_groundtruth_curvature_heatmap_overview(
            os.path.join(vis_dir, 'groundtruth_curvature_heatmap_overview.png'))

    def _save_metric_overview(self, save_path: str):
        x_values = self._sample_flat_values('curvature_gt_map')
        y_values = self._sample_flat_values('proxy_curvature_map')
        ratios = [record.mean_ratio for record in self.records]
        corrs = [record.corr for record in self.records]

        fig = plt.figure(figsize=(13, 8), dpi=150)
        gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.4])

        ax_scatter = fig.add_subplot(gs[0, 0])
        if x_values.size > 0 and y_values.size > 0:
            max_points = min(3000, x_values.size)
            indices = np.linspace(0, x_values.size - 1, max_points).astype(int)
            ax_scatter.scatter(x_values[indices], y_values[indices], s=4, alpha=0.35, c='#4ea3ff')
            bound = max(float(np.max(x_values)), float(np.max(y_values)), 1e-8) * 1.05
            ax_scatter.plot([0, bound], [0, bound], linestyle='--', linewidth=1.5, color='#f0ad4e')
            ax_scatter.set_xlim(0, bound)
            ax_scatter.set_ylim(0, bound)
        ax_scatter.set_title('Proxy Curvature vs Groundtruth Curvature (Def. 3.1 approx)')
        ax_scatter.set_xlabel('groundtruth curvature')
        ax_scatter.set_ylabel('Proxy curvature')
        ax_scatter.grid(alpha=0.25)

        ax_ratio = fig.add_subplot(gs[0, 1])
        if ratios:
            xs = np.arange(len(ratios))
            ax_ratio.plot(xs, ratios, marker='o', linewidth=2.0, color='#4ea3ff')
            ax_ratio.axhline(1.0, linestyle='--', linewidth=1.5, color='#f0ad4e', label='target=1')
            ax_ratio.set_xlim(-0.2, len(ratios) - 0.8 if len(ratios) > 1 else 0.2)
        ax_ratio.set_title('Mean Ratio per Anchor')
        ax_ratio.set_xlabel('Anchor index')
        ax_ratio.set_ylabel('mean ratio')
        ax_ratio.grid(alpha=0.25)
        ax_ratio.legend(loc='best', fontsize=8)

        ax_text = fig.add_subplot(gs[1, :])
        ax_text.axis('off')
        summary = [
            f'samples={len(self.records)}',
            f'mean corr={np.mean(corrs):.6f}' if corrs else 'mean corr=nan',
            f'mean ratio={np.mean(ratios):.6f}' if ratios else 'mean ratio=nan',
            f'std ratio={np.std(ratios):.6f}' if ratios else 'std ratio=nan',
            f'mean proxy curvature={np.mean([r.mean_proxy_curvature for r in self.records]):.6e}' if self.records else 'mean proxy curvature=nan',
            f'mean groundtruth curvature={np.mean([r.mean_curvature_gt for r in self.records]):.6e}' if self.records else 'mean groundtruth curvature=nan',
        ]
        ax_text.text(0.01, 0.8, ' | '.join(summary), fontsize=10)

        fig.suptitle('Straightness Evaluation Overview', fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(save_path)
        plt.close(fig)

    def _save_groundtruth_curvature_heatmap_overview(self, save_path: str):
        maps = [record.curvature_gt_map.mean(dim=0).detach().cpu().numpy() for record in self.records]
        labels = [f'step {record.step:04d}  t={record.t:.3f}' for record in self.records]
        if not maps:
            return

        count = len(maps)
        cols = min(3, count)
        rows = (count + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows), dpi=150)
        axes = np.array(axes).reshape(rows, cols)

        for idx in range(rows * cols):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]
            if idx >= count:
                ax.axis('off')
                continue
            im = ax.imshow(maps[idx], cmap='viridis', aspect='auto')
            ax.set_title(labels[idx], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle('Groundtruth Curvature Heatmap Overview (Def. 3.1 approx)', fontsize=14)
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cbar.set_label('groundtruth curvature')
        fig.tight_layout(rect=(0, 0, 0.98, 0.96))
        fig.savefig(save_path)
        plt.close(fig)

    def _save_heatmap_overview(self, save_path: str):
        maps = [record.straightness_map.mean(dim=0).detach().cpu().numpy() for record in self.records]
        labels = [f'step {record.step:04d}  t={record.t:.3f}' for record in self.records]
        if not maps:
            return

        count = len(maps)
        cols = min(3, count)
        rows = (count + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows), dpi=150)
        axes = np.array(axes).reshape(rows, cols)

        for idx in range(rows * cols):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]
            if idx >= count:
                ax.axis('off')
                continue
            # Per-panel normalization for clearer local contrast.
            im = ax.imshow(maps[idx], cmap='magma', aspect='auto')
            ax.set_title(labels[idx], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle('Straightness Heatmap Overview (mean over token-time)', fontsize=14)
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cbar.set_label('straightness')
        fig.tight_layout(rect=(0, 0, 0.98, 0.96))
        fig.savefig(save_path)
        plt.close(fig)

    def _sample_flat_values(self, field_name: str) -> np.ndarray:
        arrays = []
        for record in self.records:
            tensor = getattr(record, field_name)
            arrays.append(tensor.detach().cpu().reshape(-1).numpy())
        if not arrays:
            return np.asarray([], dtype=np.float32)
        return np.concatenate(arrays, axis=0)

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
    def _clear_directory(path: str):
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                continue
            os.remove(full_path)

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