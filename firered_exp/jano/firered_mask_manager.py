"""
FireRed Jano MaskManager

FireRed 的序列由两部分 token 拼接而成：
  - 噪声 latent tokens (前 noise_seq_len 个)：遵循 Jano 三级动态掩码 (static/medium/active)
  - 参考图 latent tokens (后 ref_seq_len 个)：通过一次 FFT 复杂度分析，判断区间重要性

掩码等级 (step_level)：
  0 - warmup/cooldown: 全量计算，不更新缓存
  3 - full+update: 全量计算并更新所有 token 的输出缓存
  2 - medium: 计算 medium+active 噪声 token + active 参考图 token，静态 token 从缓存恢复
  1 - active: 仅计算 active 噪声 token + active 参考图 token

输出缓存 (per branch，适配 CFG 的 cond/uncond 两路)：
  full_output_cache[branch_key] = [B, total_img_seq_len, hidden_dim] 上一次全量或稀疏推理的完整输出
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_RESET  = '\033[0m'
_BOLD   = '\033[1m'
_GRAY   = '\033[90m'
_GREEN  = '\033[32m'
_YELLOW = '\033[33m'
_CYAN   = '\033[36m'
_LEVEL_COLOR = {0: _GRAY, 1: _GREEN, 2: _YELLOW, 3: _CYAN}


def _tqdm_write(msg: str) -> None:
    try:
        from tqdm import tqdm as _tqdm
        _tqdm.write(msg)
    except Exception:
        print(msg, flush=True)


def min_max_normalize(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


# ---------------------------------------------------------------------------
# FFT 复杂度分析 — 参考图 latent
# ---------------------------------------------------------------------------

def compute_ref_fft_score(
    ref_packed: torch.Tensor,
    ref_h: int,
    ref_w: int,
    block_h: int,
    block_w: int,
) -> torch.Tensor:
    """
    对参考图 latent 做一次 2D-FFT 空间复杂度分析，返回每个 spatial block 的能量分数。

    Args:
        ref_packed: [B, ref_seq_len, C]  参考图 token（patchified, C=in_channels）
        ref_h, ref_w: 参考图 latent 的空间分辨率
        block_h, block_w: 分析 block 尺寸（与噪声 latent 的 analyze_block_size[1:] 一致）

    Returns:
        energy_score: [total_blocks] 归一化 FFT 能量分数
    """
    B, ref_seq_len, C = ref_packed.shape
    assert ref_seq_len == ref_h * ref_w, (
        f"ref_seq_len {ref_seq_len} != ref_h*ref_w {ref_h}*{ref_w}"
    )

    # 重塑为空间格式 [C, H, W]，取 batch 0
    ref_spatial = ref_packed[0].reshape(ref_h, ref_w, C).permute(2, 0, 1).float()

    # 对每个通道做 2D FFT，计算幅值谱
    fft2 = torch.fft.rfft2(ref_spatial)          # [C, H, W//2+1]
    fft_mag = fft2.abs()                          # 幅值
    # 通道平均
    spatial_mag = fft_mag.mean(dim=0)             # [H, W//2+1]

    # Pad 使 H, W 都能被 block 整除
    pad_h = (block_h - ref_h % block_h) % block_h
    pad_w_full = (block_w - ref_w % block_w) % block_w
    # W 方向在 rfft 后只有 W//2+1，pad 不超过 spatial_mag 宽度
    # 改为在原始空间分块，用 unfold
    # 直接 unfold ref_spatial 原始信号
    ref_padded = F.pad(ref_spatial, (0, pad_w_full, 0, pad_h))  # [C, H', W']
    C_, H_, W_ = ref_padded.shape
    nh = H_ // block_h
    nw = W_ // block_w
    total_blocks = nh * nw

    # 分块 [C, nh, block_h, nw, block_w] → [total_blocks, C, block_h, block_w]
    ref_blocks = ref_padded.reshape(C_, nh, block_h, nw, block_w)
    ref_blocks = ref_blocks.permute(1, 3, 0, 2, 4)             # [nh, nw, C, block_h, block_w]
    ref_blocks = ref_blocks.reshape(total_blocks, C_, block_h, block_w)

    # 每个 block 的 FFT 能量 = 高频分量的 L2 范数
    fft_b = torch.fft.rfft2(ref_blocks)           # [total_blocks, C, block_h, block_w//2+1]
    energy = (fft_b.abs() ** 2).mean(dim=(1, 2, 3))  # [total_blocks]

    return min_max_normalize(energy), nh, nw


def generate_ref_sequence_mask(
    ref_packed: torch.Tensor,
    ref_h: int,
    ref_w: int,
    block_h: int,
    block_w: int,
    ref_thresh: float,
) -> torch.Tensor:
    """
    生成参考图 token 的二级 sequence mask。
    值为 1 (static) 或 3 (active)，不设 medium 级。

    Args:
        ref_thresh: FFT 能量阈值，超过则视为 active

    Returns:
        ref_seq_mask: [ref_seq_len] int8 tensor，1=static, 3=active
    """
    energy, nh, nw = compute_ref_fft_score(ref_packed, ref_h, ref_w, block_h, block_w)

    block_mask = torch.where(energy > ref_thresh,
                             torch.tensor(3, dtype=torch.int8),
                             torch.tensor(1, dtype=torch.int8))  # [total_blocks]
    block_mask_3d = block_mask.reshape(nh, nw)

    # 展开到完整分辨率
    latent_mask = torch.zeros(ref_h, ref_w, dtype=torch.int8)
    for bi in range(nh):
        for bj in range(nw):
            r0, r1 = bi * block_h, min((bi + 1) * block_h, ref_h)
            c0, c1 = bj * block_w, min((bj + 1) * block_w, ref_w)
            latent_mask[r0:r1, c0:c1] = block_mask_3d[bi, bj]

    seq_mask = latent_mask.flatten()  # [ref_seq_len]

    active_pct = (seq_mask == 3).float().mean().item() * 100
    static_pct = (seq_mask == 1).float().mean().item() * 100
    _tqdm_write(
        f"{_BOLD}[Ref FFT Mask]{_RESET}  "
        f"{_GRAY}static:{_RESET}{static_pct:5.1f}%  "
        f"{_GREEN}active:{_RESET}{active_pct:5.1f}%"
    )
    return seq_mask


# ---------------------------------------------------------------------------
# FireRedMaskManager
# ---------------------------------------------------------------------------

class FireRedMaskManager:
    """
    管理 FireRed 噪声 + 参考图两部分 token 的掩码与缓存。

    img sequence layout:  [noise_tokens (0..noise_seq_len-1), ref_tokens (noise_seq_len..)]
    """

    def __init__(
        self,
        noise_seq_len: int,
        ref_seq_len: int,
        num_inference_steps: int,
        warmup_steps: int,
        cooldown_steps: int,
        static_interval: int,
        medium_interval: int,
    ):
        self.noise_seq_len = noise_seq_len
        self.ref_seq_len = ref_seq_len
        self.total_img_seq_len = noise_seq_len + ref_seq_len

        self.num_inference_steps = num_inference_steps
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.static_interval = static_interval
        self.medium_interval = medium_interval

        # step level (0/1/2/3)
        self.step_level: int = 0

        # 噪声 token 三级掩码（由 DynamicAnalyzer 产生）
        self.noise_static_mask: torch.Tensor | None = None   # [noise_seq_len] bool
        self.noise_medium_mask: torch.Tensor | None = None
        self.noise_active_mask: torch.Tensor | None = None

        # 参考图 token 二级掩码（FFT 一次分析产生）
        self.ref_static_mask: torch.Tensor | None = None     # [ref_seq_len] bool
        self.ref_active_mask: torch.Tensor | None = None

        # 合并掩码缓存（避免每步重复拼接）
        self._compute_mask_l1: torch.Tensor | None = None    # level-1 compute mask
        self._compute_mask_l2: torch.Tensor | None = None    # level-2 compute mask
        # 只缓存 sel_img_freqs（img 选择结果，两 CFG 路相同）
        # txt_freqs 不缓存，因为 cond/uncond 两路文本长度可能不同
        self._sel_img_freqs_l1: torch.Tensor | None = None
        self._sel_img_freqs_l2: torch.Tensor | None = None

        self.masks_ready: bool = False

        # per-branch 输出缓存 {branch_key: [B, total_img_seq_len, D]}
        self.output_cache: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ setup

    def set_noise_sequence_mask(self, noise_seq_mask: torch.Tensor) -> None:
        """
        接收 DynamicAnalyzer 产生的噪声 token sequence mask。
        Args:
            noise_seq_mask: [noise_seq_len] int8，值为 1/2/3
        """
        dev = noise_seq_mask.device
        self.noise_static_mask = (noise_seq_mask == 1).to(dev)
        self.noise_medium_mask = (noise_seq_mask == 2).to(dev)
        self.noise_active_mask = (noise_seq_mask == 3).to(dev)

        n_s = self.noise_static_mask.sum().item()
        n_m = self.noise_medium_mask.sum().item()
        n_a = self.noise_active_mask.sum().item()
        _tqdm_write(
            f"{_BOLD}[Noise Mask]{_RESET}  "
            f"{_GRAY}static={n_s}{_RESET}  "
            f"{_YELLOW}medium={n_m}{_RESET}  "
            f"{_GREEN}active={n_a}{_RESET}  "
            f"total={self.noise_seq_len}"
        )

    def set_ref_sequence_mask(self, ref_seq_mask: torch.Tensor) -> None:
        """
        接收 FFT 分析产生的参考图 token sequence mask。
        Args:
            ref_seq_mask: [ref_seq_len] int8，值为 1/3
        """
        dev = ref_seq_mask.device
        self.ref_static_mask = (ref_seq_mask == 1).to(dev)
        self.ref_active_mask = (ref_seq_mask == 3).to(dev)
        self.masks_ready = True
        # 清除合并掩码缓存
        self._compute_mask_l1 = None
        self._compute_mask_l2 = None
        self._sel_img_freqs_l1 = None
        self._sel_img_freqs_l2 = None

    # ------------------------------------------------------------------ level

    def update_step_level(self, step_index: int) -> None:
        """
        根据 step_index（0-based）更新 step_level。
        step_index 对应 PAB/ToCa 的 step_index，与 Jano 的 timestep 偏移 1。
        """
        # 以 1-based timestep 对齐 Jano 的 update_step_level 逻辑
        timestep = step_index + 1

        if (
            not self.masks_ready
            or timestep <= self.warmup_steps
            or timestep > self.num_inference_steps - self.cooldown_steps
            or self.static_interval * self.medium_interval == 0
        ):
            self.step_level = 0
        elif (timestep - self.warmup_steps - 1) % self.static_interval == 0:
            self.step_level = 3
        elif (timestep - self.warmup_steps - 1) % self.medium_interval == 0:
            self.step_level = 2
        else:
            self.step_level = 1

        lc = _LEVEL_COLOR[self.step_level]
        alloc_gb = torch.cuda.memory_allocated() / 1024 ** 3

        if self.total_img_seq_len:
            sel = self._get_num_selected()
            seq_pct = sel / self.total_img_seq_len * 100
        else:
            seq_pct = 100.0

        _tqdm_write(
            f"{lc}[L{self.step_level}]{_RESET} "
            f"step={step_index}  seq={seq_pct:.0f}%  GPU={alloc_gb:.1f}GB"
        )

    def _get_num_selected(self) -> int:
        if self.step_level in (0, 3):
            return self.total_img_seq_len
        mask = self.get_compute_mask()
        return int(mask.sum().item()) if mask is not None else self.total_img_seq_len

    # ------------------------------------------------------------------ masks

    def get_compute_mask(self) -> torch.Tensor | None:
        """
        返回 [total_img_seq_len] bool 掩码，True 表示该 token 需本步计算。
        level 0/3: None（全量，不筛选）
        level 2: medium+active 噪声 + active 参考图
        level 1: active 噪声 + active 参考图
        """
        if self.step_level in (0, 3) or not self.masks_ready:
            return None

        if self.step_level == 2:
            if self._compute_mask_l2 is None:
                noise_part = ~self.noise_static_mask
                ref_part = self.ref_active_mask
                self._compute_mask_l2 = torch.cat([noise_part, ref_part])
            return self._compute_mask_l2

        # level 1
        if self._compute_mask_l1 is None:
            noise_part = self.noise_active_mask
            ref_part = self.ref_active_mask
            self._compute_mask_l1 = torch.cat([noise_part, ref_part])
        return self._compute_mask_l1

    def get_masked_rotary(
        self,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        txt_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对 image_rotary_emb = (img_freqs, txt_freqs) 按 compute_mask 做筛选。

        FireRed 的 rotary emb 由两个独立张量组成：
          - img_freqs: [img_seq_len, head_dim//2]  仅作用于图像流 QK（做 token 选择）
          - txt_freqs: [txt_seq_len, head_dim//2]  仅作用于文本流 QK（原样返回）

        注意：
          - sel_img_freqs 会被缓存（cond/uncond 两路 img layout 相同）
          - txt_freqs 不缓存，每次使用调用方传入的新值；
            cond 与 uncond 的文本序列长度可能不同，缓存会导致 shape 错误。
        """
        if self.step_level in (0, 3) or not self.masks_ready:
            return image_rotary_emb

        img_freqs, txt_freqs = image_rotary_emb

        cache_attr = "_sel_img_freqs_l2" if self.step_level == 2 else "_sel_img_freqs_l1"
        sel_img_freqs = getattr(self, cache_attr)
        if sel_img_freqs is None:
            compute_mask = self.get_compute_mask()
            mask = compute_mask.to(img_freqs.device)
            sel_img_freqs = img_freqs[mask]      # [num_selected, head_dim//2]
            setattr(self, cache_attr, sel_img_freqs)

        return (sel_img_freqs, txt_freqs)        # txt_freqs 始终来自当前调用

    # ------------------------------------------------------------------ cache

    def store_output(self, output: torch.Tensor, branch_key: str) -> None:
        """
        在 level-3 步结束后，将完整输出缓存（用于后续 level-2/1 恢复）。
        Args:
            output: [B, total_img_seq_len, D] transformer block loop 后的隐状态
        """
        self.output_cache[branch_key] = output.detach()

    def restore_output(
        self,
        selected_output: torch.Tensor,
        compute_mask: torch.Tensor,
        branch_key: str,
    ) -> torch.Tensor:
        """
        将稀疏计算结果合并回完整序列，静态 token 从缓存中恢复。

        Args:
            selected_output: [B, num_selected, D]
            compute_mask: [total_img_seq_len] bool
            branch_key: "branch_0" / "branch_1"

        Returns:
            full_output: [B, total_img_seq_len, D]
        """
        cached = self.output_cache.get(branch_key)
        assert cached is not None, (
            f"[FireRedMaskManager] No cached output for {branch_key}. "
            "Ensure a level-3 step has run before level-1/2."
        )

        dev = selected_output.device
        full_output = cached.to(dev).clone()

        # 将本步计算结果写入对应位置（自动覆盖 medium/active 位置）
        full_output[:, compute_mask.to(dev), :] = selected_output

        # level-2 时：同步更新缓存中的 medium 噪声 token（为 level-1 准备）
        if self.step_level == 2:
            self.output_cache[branch_key] = full_output.detach()

        return full_output

    def clear_cache(self) -> None:
        """清空所有缓存（新一轮推理前调用）。"""
        self.output_cache.clear()
        self._compute_mask_l1 = None
        self._compute_mask_l2 = None
        self._sel_img_freqs_l1 = None
        self._sel_img_freqs_l2 = None
        self.masks_ready = False
        self.noise_static_mask = None
        self.noise_medium_mask = None
        self.noise_active_mask = None
        self.ref_static_mask = None
        self.ref_active_mask = None
        self.step_level = 0
