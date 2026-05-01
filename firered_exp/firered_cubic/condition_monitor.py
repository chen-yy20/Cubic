from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


class ConditionComplexityMonitor:
    """Estimate condition-token spatial complexity from condition image latents.

    This monitor is intentionally scheduler-independent: it derives a static
    token-space complexity map from encoded condition latents (for example via
    FFT high-frequency energy) and reuses it for cond partition/tau planning.
    """

    def __init__(self, config):
        self.config = config
        self.complexity_map: Optional[torch.Tensor] = None
        self.complexity_history = []

    def reset(self) -> None:
        self.complexity_map = None
        self.complexity_history = []

    def build_from_condition_latents(
        self,
        cond_latents: torch.Tensor,
        step_idx: int | None = None,
    ) -> torch.Tensor:
        """Build token-space complexity map from unpacked condition latents.

        Args:
            cond_latents: Tensor with shape (B, C, F, H, W).
                FireRed i2i condition latents are expected to have F=1.
        Returns:
            Token-space complexity map with shape (Ft, Ht, Wt).
        """
        if cond_latents.ndim != 5:
            raise ValueError(f"cond_latents must be 5D (B,C,F,H,W), got shape={tuple(cond_latents.shape)}")

        x = cond_latents.float()
        # Collapse temporal axis for image-edit condition (F is typically 1).
        x2d = x.mean(dim=2)

        # Remove low-frequency bias before FFT.
        x2d = x2d - x2d.mean(dim=(-2, -1), keepdim=True)
        fft = torch.fft.fft2(x2d, dim=(-2, -1), norm="ortho")

        h, w = x2d.shape[-2:]
        yy = torch.fft.fftfreq(h, d=1.0, device=x2d.device).view(h, 1)
        xx = torch.fft.fftfreq(w, d=1.0, device=x2d.device).view(1, w)
        radius = torch.sqrt(xx * xx + yy * yy)
        radius = radius / max(float(radius.max().item()), 1e-8)

        # FFT-based high-pass emphasis: larger radius => larger weight.
        hp_fft = fft * radius.view(1, 1, h, w)
        hp = torch.fft.ifft2(hp_fft, dim=(-2, -1), norm="ortho").real

        # Spatial complexity proxy in latent grid.
        complexity_hw = hp.abs().mean(dim=(0, 1))
        complexity_map = self._pool_to_token_space(complexity_hw)

        self.complexity_map = complexity_map
        self.complexity_history.append(
            {
                "step": int(step_idx) if step_idx is not None else None,
                "t": 0.0,
                "curvature_map": complexity_map.detach().cpu(),
            }
        )
        return complexity_map

    def get_block_curvatures(self, partition, complexity_map: Optional[torch.Tensor] = None) -> Dict[int, float]:
        comp_map = self.complexity_map if complexity_map is None else complexity_map
        if comp_map is None:
            return {}

        q = float(getattr(self.config, "condition_complexity_percentile", 0.7))
        q = min(max(q, 0.0), 1.0)

        comp_np = comp_map.detach().float().cpu().numpy()
        block_scores: Dict[int, float] = {}
        for block in partition:
            region = comp_np[
                block.f_start:block.f_end,
                block.h_start:block.h_end,
                block.w_start:block.w_end,
            ]
            if region.size == 0:
                block_scores[block.block_id] = 0.0
                continue
            block_scores[block.block_id] = float(np.quantile(region.reshape(-1), q))
        return block_scores

    def _pool_to_token_space(self, complexity_hw: torch.Tensor) -> torch.Tensor:
        pt, ph, pw = getattr(self.config, "patch_size", (1, 2, 2))
        pt, ph, pw = int(pt), int(ph), int(pw)
        if pt <= 0 or ph <= 0 or pw <= 0:
            raise ValueError("patch_size must be positive")

        if pt != 1:
            raise ValueError(f"ConditionComplexityMonitor expects pt=1 for image condition, got pt={pt}")

        h_dim, w_dim = complexity_hw.shape
        h_eff = (h_dim // ph) * ph
        w_eff = (w_dim // pw) * pw
        if h_eff == 0 or w_eff == 0:
            raise ValueError("condition complexity map is smaller than patch_size")

        x = complexity_hw[:h_eff, :w_eff].unsqueeze(0).unsqueeze(0)
        pooled = F.avg_pool2d(x, kernel_size=(ph, pw), stride=(ph, pw))
        return pooled.squeeze(0).squeeze(0).unsqueeze(0)
