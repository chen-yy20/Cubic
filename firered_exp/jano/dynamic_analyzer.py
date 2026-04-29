"""
DynamicAnalyzer — 对 warmup 阶段收集的噪声 latent 进行时空动态性分析。

原始实现来自 Jano 项目，此处内化为 Cubic/firered_exp/jano 包的一部分，
不再依赖外部 Jano 项目路径。

依赖（均为 Cubic 内部模块或标准库）：
  - utils.envs.GlobalEnv
  - utils.timer.get_timer
  - scipy, skimage（标准科学计算库，用于 _enhance_combined_score）
"""

from __future__ import annotations

import torch
import torch.fft as fft
import numpy as np
from scipy import ndimage
from skimage import morphology, measure

from utils.envs import GlobalEnv
from utils.timer import get_timer

from .block_manager import init_block_manager

# ── Terminal colour helpers ──────────────────────────────────────────────────
_RESET = "\033[0m"
_BOLD  = "\033[1m"
_CYAN  = "\033[36m"
_GRAY  = "\033[90m"
_GREEN = "\033[32m"


def _tqdm_write(msg: str) -> None:
    try:
        from tqdm import tqdm as _tqdm
        _tqdm.write(msg)
    except Exception:
        print(msg, flush=True)


def min_max_normalize(data: torch.Tensor) -> torch.Tensor:
    lo, hi = data.min(), data.max()
    if hi - lo != 0:
        return (data - lo) / (hi - lo)
    return torch.zeros_like(data)


class DynamicAnalyzer:
    """
    收集 warmup 阶段的噪声 latent 特征，分析时空动态性，
    输出各 block 的 enhanced_combined_score（归一化到 [0,1]）。

    用法（在 FireRed Jano 适配中）：
        da = DynamicAnalyzer(T=1, H=noise_h, W=noise_w, C=in_channels)
        # 在 warmup 阶段，每步直接追加：
        da.stored_features.append(noise_spatial)   # [C, 1, H, W]
        # warmup 结束后：
        enhanced_combined = da.analyze()
        da.stored_features = None  # 阻止后续追加
    """

    def __init__(self, T: int, H: int, W: int, C: int):
        self.block_manager = init_block_manager(T, H, W, C)
        self.analysis_steps = GlobalEnv.get_envs("warmup_steps")
        self.stored_features: list | None = []
        self.tag = GlobalEnv.get_envs("tag")

        self.temporal_weight  = GlobalEnv.get_envs("t_weight")
        self.spatial_weight   = 1.0 - self.temporal_weight
        self.diffusion_strength      = GlobalEnv.get_envs("d_strength")
        self.max_diffusion_distance  = GlobalEnv.get_envs("d_distance")

        _tqdm_write(
            f"\n{_BOLD}{_CYAN}[DynamicAnalyzer]{_RESET} tag={self.tag}  "
            f"T={T} H={H} W={W} C={C}  "
            f"blocks=(nt={self.block_manager.nt}, "
            f"nh={self.block_manager.nh}, "
            f"nw={self.block_manager.nw})  "
            f"warmup={self.analysis_steps} steps"
        )

        self.block_mask:  torch.Tensor | None = None
        self.latent_mask: torch.Tensor | None = None

    # ------------------------------------------------------------------ dynamics

    def _compute_dynamics(
        self, blocked_latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算时空动态性（全向量化版）。

        blocked_latents: [steps, total_blocks, bt, bh*bw, C]

        Returns:
            temporal_dynamics: [total_blocks]
            spatial_dynamics:  [total_blocks]
        """
        steps, block_num, t, s, C = blocked_latents.shape
        device   = blocked_latents.device
        interval = steps // 2

        def _pairwise_l2(x: torch.Tensor) -> torch.Tensor:
            """x: (..., n, C) → pairwise L2 (..., n, n)"""
            x_sq  = (x * x).sum(dim=-1)
            cross = torch.matmul(x, x.transpose(-1, -2))
            return (x_sq.unsqueeze(-1) + x_sq.unsqueeze(-2) - 2.0 * cross).clamp_(min=0.0).sqrt_()

        # 时间动态性
        if self.block_manager.T > 1 and interval > 0:
            xt     = blocked_latents.permute(1, 3, 0, 2, 4).contiguous()  # [B,s,steps,t,C]
            dm_t   = _pairwise_l2(xt)
            delta_t = (dm_t[:, :, interval:] - dm_t[:, :, :steps - interval]).abs_()
            temporal_dynamics = delta_t.mean(dim=(2, 3, 4)).mean(dim=1)
        else:
            temporal_dynamics = torch.zeros(block_num, device=device)

        # 空间动态性
        if interval > 0:
            xs     = blocked_latents.permute(1, 2, 0, 3, 4).contiguous()  # [B,t,steps,s,C]
            dm_s   = _pairwise_l2(xs)
            delta_s = (dm_s[:, :, interval:] - dm_s[:, :, :steps - interval]).abs_()
            spatial_dynamics = delta_s.mean(dim=(2, 3, 4)).mean(dim=1)
        else:
            spatial_dynamics = torch.zeros(block_num, device=device)

        return temporal_dynamics, spatial_dynamics

    def _enhance_combined_score(
        self,
        original_combined: torch.Tensor,
        block_dims: tuple[int, int, int],
    ) -> torch.Tensor:
        """向内扩散增强 combined score，使轮廓内部的重要区域得到更高权重。"""
        nt, nh, nw = block_dims
        enhanced_combined = original_combined.clone()

        for t in range(nt):
            current_layer = original_combined.reshape(nt, nh, nw)[t, :, :]

            score_std  = torch.std(current_layer)
            score_mean = torch.mean(current_layer)

            contour_threshold  = score_mean + 0.3 * score_std
            contour_mask       = current_layer > contour_threshold
            internal_threshold = score_mean
            potential_internal = current_layer < internal_threshold

            if not torch.any(contour_mask) or not torch.any(potential_internal):
                continue

            # 边界禁止区域
            forbidden = torch.zeros_like(current_layer, dtype=torch.bool)
            forbidden[0, :] = True
            forbidden[-1, :] = True
            forbidden[:, 0] = True
            forbidden[:, -1] = True

            candidate_internal = potential_internal & (~forbidden)

            labeled = torch.from_numpy(
                measure.label(candidate_internal.cpu().numpy())
            ).to(current_layer.device)

            true_internal = torch.zeros_like(current_layer, dtype=torch.bool)

            for region_id in range(1, int(labeled.max().item()) + 1):
                region_mask = labeled == region_id
                dilated      = morphology.binary_dilation(region_mask.cpu().numpy())
                boundary     = torch.from_numpy(dilated & (~region_mask.cpu().numpy())).to(current_layer.device)

                if torch.any(contour_mask):
                    dist_to_contour = torch.from_numpy(
                        ndimage.distance_transform_edt(~contour_mask.cpu().numpy())
                    ).to(current_layer.device)
                    bd_dists = dist_to_contour[boundary]
                    close_ratio = torch.mean((bd_dists <= 2.0).float())
                    if close_ratio >= 0.3 and torch.sum(region_mask) >= 2:
                        true_internal |= region_mask

            if torch.any(true_internal) and torch.any(contour_mask):
                dist_from_contour = torch.from_numpy(
                    ndimage.distance_transform_edt(~contour_mask.cpu().numpy())
                ).to(current_layer.device)

                valid_diff = true_internal & (dist_from_contour <= self.max_diffusion_distance)
                if torch.any(valid_diff):
                    dists  = dist_from_contour[valid_diff]
                    weights = torch.exp(-dists / (self.max_diffusion_distance / 2))
                    avg_contour = torch.mean(current_layer[contour_mask])
                    enhancements = weights * avg_contour * self.diffusion_strength

                    enhanced_layer = current_layer.clone()
                    enhanced_layer[valid_diff] += enhancements

                    s = t * nh * nw
                    enhanced_combined[s : s + nh * nw] = enhanced_layer.flatten()

        return enhanced_combined

    # ------------------------------------------------------------------ analyze

    @get_timer("analyze")
    def analyze(self) -> torch.Tensor | None:
        """
        利用已收集的 stored_features 计算时空动态性，返回 enhanced_combined_score。

        Returns:
            enhanced_combined: [total_blocks] float tensor（归一化到 [0,1]），
            若无数据则返回 None。
        """
        if not self.stored_features:
            _tqdm_write(f"{_GRAY}[Analyze] No features stored.{_RESET}")
            return None

        _tqdm_write(
            f"\n{_BOLD}{_CYAN}[Analyze]{_RESET} "
            f"Running dynamics analysis for tag='{self.tag}' ..."
        )

        latents_tensor  = torch.stack(self.stored_features, dim=0).cuda()
        blocked_latents = self.block_manager.block_3d(latents_tensor)

        temporal_dynamics, spatial_dynamics = self._compute_dynamics(blocked_latents)

        original_combined = (
            self.temporal_weight * temporal_dynamics
            + self.spatial_weight * spatial_dynamics
        )
        original_combined  = min_max_normalize(original_combined)
        enhanced_combined  = self._enhance_combined_score(
            original_combined,
            (self.block_manager.nt, self.block_manager.nh, self.block_manager.nw),
        )

        _tqdm_write(f"{_GREEN}[Analyze]{_RESET} Done — tag='{self.tag}'\n")
        return enhanced_combined
