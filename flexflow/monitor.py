"""Curvature monitor for Cubic anchor steps.

This module follows flexflow/docs/implement.md section 4.4:
- endpoint prediction update at anchor step
- drift-based curvature estimation in latent space
- pooling to token space with patch_size
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class CurvatureMonitor:
    """Track endpoint drift and estimate token-space curvature map."""

    def __init__(self, config):
        self.config = config
        self.prev_endpoint: Optional[torch.Tensor] = None
        self.prev_t: Optional[float] = None
        self.curvature_map: Optional[torch.Tensor] = None
        self.curvature_history = []

    def update(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        t: float | torch.Tensor,
        step_idx: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Update curvature state at anchor step.

        Args:
            x: Current state, shape (B, C, F, H, W).
            v: Current velocity, shape (B, C, F, H, W).
            t: Current scalar timestep.

        Returns:
            Token-space curvature map (Ft, Ht, Wt), or None on first call.
        """
        endpoint = x + (1.0 - self._to_scalar_t(t)) * v

        if self.prev_endpoint is not None and self.prev_t is not None:
            drift = endpoint - self.prev_endpoint
            drift_norm = torch.linalg.vector_norm(drift, ord=2, dim=1)

            t_cur = self._to_scalar_t(t)
            eps = 1e-8
            normalizer = (1.0 - t_cur) * abs(t_cur - float(self.prev_t)) + eps
            curvature_latent = drift_norm / normalizer

            curvature_fhw = curvature_latent.mean(dim=0)
            self.curvature_map = self._pool_to_token_space(curvature_fhw)
            self.curvature_history.append({
                'step': int(step_idx) if step_idx is not None else None,
                't': float(t_cur),
                'curvature_map': self.curvature_map.detach().cpu(),
            })

        self.prev_endpoint = endpoint.detach()
        self.prev_t = self._to_scalar_t(t)
        return self.curvature_map

    def get_block_curvatures(self, partition) -> Dict[int, float]:
        """Aggregate token curvature to block-level curvature (percentile)."""
        if self.curvature_map is None:
            return {}

        q = float(getattr(self.config, "curvature_percentile", 0.9))
        q = min(max(q, 0.0), 1.0)

        block_curvatures: Dict[int, float] = {}
        for block in partition:
            region = self.curvature_map[
                block.f_start:block.f_end,
                block.h_start:block.h_end,
                block.w_start:block.w_end,
            ]
            if region.numel() == 0:
                block_curvatures[block.block_id] = 0.0
                continue
            block_curvatures[block.block_id] = float(
                torch.quantile(region.float().reshape(-1), q).item())

        return block_curvatures

    def reset(self):
        """Reset monitor state before a new sampling run."""
        self.prev_endpoint = None
        self.prev_t = None
        self.curvature_map = None
        self.curvature_history = []

    def export_visualizations(self,
                              output_dir: str,
                              max_frames: int = 8,
                              save_raw: bool = False,
                              save_strip: bool = False,
                              save_meta: bool = False):
        """Export anchor-step curvature maps to results directory.

        Output structure:
            output_dir/curvature_vis/
              - curvature_overview.png

        Optional, default-off outputs:
              - step_XXXX_meta.txt
              - step_XXXX_map.npz
              - step_XXXX_mean_hw.png
              - step_XXXX_time_strip.png
        """
        if not self.curvature_history:
            return

        vis_dir = os.path.join(output_dir, 'curvature_vis')
        os.makedirs(vis_dir, exist_ok=True)
        self._clear_directory(vis_dir)

        mean_tiles = []
        labels = []

        for item in self.curvature_history:
            step = item['step'] if item['step'] is not None else -1
            t_val = float(item['t'])
            curv = item['curvature_map'].float()
            ft, ht, wt = curv.shape

            stem = f"step_{step:04d}" if step >= 0 else "step_unknown"
            mean_hw = curv.mean(dim=0)
            mean_tiles.append(mean_hw.detach().cpu().numpy())
            labels.append(f"step {step:04d}  t={t_val:.3f}")

            if save_raw:
                npz_path = os.path.join(vis_dir, f"{stem}_map.npz")
                np.savez_compressed(npz_path, curvature_map=curv.numpy())

            if save_meta:
                meta_path = os.path.join(vis_dir, f"{stem}_meta.txt")
                with open(meta_path, 'w', encoding='utf-8') as f:
                    f.write(f"step={step}\n")
                    f.write(f"t={t_val:.8f}\n")
                    f.write(f"shape=({ft}, {ht}, {wt})\n")
                    f.write(f"mean={float(curv.mean().item()):.8e}\n")
                    f.write(f"std={float(curv.std().item()):.8e}\n")

            if save_strip:
                frame_count = min(max_frames, ft)
                frame_indices = torch.linspace(0, ft - 1, frame_count).round().long().tolist()
                tiles = [curv[idx] for idx in frame_indices]
                strip = torch.cat(tiles, dim=1)
                self._save_single_heatmap(
                    strip.detach().cpu().numpy(),
                    os.path.join(vis_dir, f"{stem}_time_strip.png"),
                    title=f"{stem} time strip",
                )

        self._save_heatmap_grid(
            mean_tiles,
            labels,
            os.path.join(vis_dir, 'curvature_overview.png'),
            suptitle='Anchor Curvature Mean-HW Overview',
            cmap='turbo',
        )

    def _save_single_heatmap(self, array_2d: np.ndarray, save_path: str, title: str):
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        im = ax.imshow(array_2d, cmap='turbo', aspect='auto')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)

    def _save_heatmap_grid(self, maps, labels, save_path: str, suptitle: str, cmap: str):
        if not maps:
            raise ValueError('no curvature maps to visualize')

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
            im = ax.imshow(maps[idx], cmap=cmap, aspect='auto')
            ax.set_title(labels[idx], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(suptitle, fontsize=14)
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cbar.set_label('curvature')
        fig.tight_layout(rect=(0.0, 0.0, 0.98, 0.96))
        fig.savefig(save_path)
        plt.close(fig)

    def _clear_directory(self, path: str):
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                continue
            os.remove(full_path)

    def _pool_to_token_space(self, curvature_fhw: torch.Tensor) -> torch.Tensor:
        """Pool latent-space curvature (F,H,W) to token-space (Ft,Ht,Wt)."""
        pt, ph, pw = getattr(self.config, "patch_size", (1, 2, 2))
        pt, ph, pw = int(pt), int(ph), int(pw)
        if pt <= 0 or ph <= 0 or pw <= 0:
            raise ValueError("patch_size must be positive")

        f_dim, h_dim, w_dim = curvature_fhw.shape
        f_eff = (f_dim // pt) * pt
        h_eff = (h_dim // ph) * ph
        w_eff = (w_dim // pw) * pw
        if f_eff == 0 or h_eff == 0 or w_eff == 0:
            raise ValueError("curvature map is smaller than patch_size")

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
                raise ValueError("empty tensor timestep")
            return float(t.detach().reshape(-1)[0].item())
        return float(t)
