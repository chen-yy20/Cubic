"""Curvature monitor for Cubic anchor steps.

This module follows cubic_core/docs/implement.md section 4.4:
- endpoint prediction update at anchor step
- drift-based curvature estimation in latent space
- pooling to token space with patch_size
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .visualizer import CubicVisualizer


class CurvatureMonitor:
    """Track endpoint drift and estimate token-space curvature map."""

    def __init__(self, config):
        self.config = config
        self.visualizer = CubicVisualizer(config)
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

    def get_block_curvatures(self, partition, curvature_map: Optional[torch.Tensor] = None) -> Dict[int, float]:
        """Aggregate token curvature to block-level curvature (percentile)."""
        curv_map = self.curvature_map if curvature_map is None else curvature_map
        if curv_map is None:
            return {}

        q = float(getattr(self.config, "curvature_percentile", 0.9))
        q = min(max(q, 0.0), 1.0)

        use_gpu_agg = bool(getattr(self.config, 'partition_gpu_aggregation', True))
        if use_gpu_agg and curv_map.device.type == 'cuda':
            selected_size = getattr(partition, 'selected_block_size', None)
            if selected_size is not None:
                quantiles = self._uniform_block_quantiles(curv_map.detach().float(), selected_size, q)
                out: Dict[int, float] = {}
                q_list = quantiles.detach().cpu().tolist()
                for idx, block in enumerate(partition):
                    out[block.block_id] = float(q_list[idx])
                return out

        # Avoid many tiny GPU quantile kernels. Materialize once on CPU and aggregate there.
        curv_np = curv_map.detach().float().cpu().numpy()

        block_curvatures: Dict[int, float] = {}
        for block in partition:
            region = curv_np[
                block.f_start:block.f_end,
                block.h_start:block.h_end,
                block.w_start:block.w_end,
            ]
            if region.size == 0:
                block_curvatures[block.block_id] = 0.0
                continue
            block_curvatures[block.block_id] = float(np.quantile(region.reshape(-1), q))

        return block_curvatures

    def _uniform_block_quantiles(
        self,
        curv_map: torch.Tensor,
        block_size: tuple[int, int, int],
        q: float,
    ) -> torch.Tensor:
        ft, ht, wt = curv_map.shape
        bf, bh, bw = block_size
        nf = (ft + bf - 1) // bf
        nh = (ht + bh - 1) // bh
        nw = (wt + bw - 1) // bw

        pf = nf * bf
        ph = nh * bh
        pw = nw * bw

        padded = torch.zeros((pf, ph, pw), dtype=curv_map.dtype, device=curv_map.device)
        padded[:ft, :ht, :wt] = curv_map

        mask = torch.zeros((pf, ph, pw), dtype=torch.bool, device=curv_map.device)
        mask[:ft, :ht, :wt] = True

        values = padded.view(nf, bf, nh, bh, nw, bw)
        values = values.permute(0, 2, 4, 1, 3, 5).reshape(nf * nh * nw, bf * bh * bw)

        valid = mask.view(nf, bf, nh, bh, nw, bw)
        valid = valid.permute(0, 2, 4, 1, 3, 5).reshape(nf * nh * nw, bf * bh * bw)

        values = values.masked_fill(~valid, torch.nan)
        quantiles = torch.nanquantile(values, q, dim=1)
        return torch.nan_to_num(quantiles, nan=0.0, posinf=0.0, neginf=0.0)

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
        """Backward-compatible entry; visualization is centralized in cubic_core.visualizer."""
        self.visualizer.export_curvature_history(
            curvature_history=self.curvature_history,
            output_dir=output_dir,
            max_frames=max_frames,
            save_raw=save_raw,
            save_strip=save_strip,
            save_meta=save_meta,
        )

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
