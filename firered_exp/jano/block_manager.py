"""
BlockManager — 将潜空间张量分成 3D 块，供 DynamicAnalyzer 使用。

原始实现来自 Jano 项目，此处内化为 Cubic/firered_exp/jano 包的一部分，
不再依赖外部 Jano 项目路径。
"""

import torch
import torch.nn.functional as F

from utils.envs import GlobalEnv


class BlockManager:
    def __init__(self, T: int, H: int, W: int, C: int):
        self.T = T
        self.H = H
        self.W = W
        self.C = C
        self.latent_shape = (C, T, H, W)
        self.block_size = GlobalEnv.get_envs("analyze_block_size")
        if self.T == 1:
            assert self.block_size[0] == 1, (
                f"Unsupported block_size {self.block_size} for T=1 latent."
            )

        bt, bh, bw = self.block_size
        self.padded_T = ((T + bt - 1) // bt) * bt
        self.padded_H = ((H + bh - 1) // bh) * bh
        self.padded_W = ((W + bw - 1) // bw) * bw

        self.nt = self.padded_T // bt
        self.nh = self.padded_H // bh
        self.nw = self.padded_W // bw
        self.total_blocks = self.nt * self.nh * self.nw

    def _pad_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        对 [step, C, T, H, W] tensor 做零填充，使 T/H/W 均为 block_size 的整数倍。
        """
        _step, _C, T, H, W = tensor.shape
        pad_T = self.padded_T - T
        pad_H = self.padded_H - H
        pad_W = self.padded_W - W
        return F.pad(tensor, (0, pad_W, 0, pad_H, 0, pad_T), mode="constant", value=0)

    def block_3d(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        将 [step, C, T, H, W] 张量分块。

        Returns:
            blocked: [step, total_blocks, bt, bh*bw, C]
        """
        padded = self._pad_tensor(tensor)
        step, C, _T, _H, _W = padded.shape
        bt, bh, bw = self.block_size

        # [step, T, H, W, C]
        x = padded.permute(0, 2, 3, 4, 1)
        # [step, nt, bt, nh, bh, nw, bw, C]
        x = x.view(step, self.nt, bt, self.nh, bh, self.nw, bw, C)
        # [step, nt, nh, nw, bt, bh, bw, C]
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        # [step, total_blocks, bt, bh*bw, C]
        x = x.reshape(step, self.total_blocks, bt, bh * bw, C)
        return x


_BM: BlockManager | None = None


def init_block_manager(T: int, H: int, W: int, C: int) -> BlockManager:
    global _BM
    _BM = BlockManager(T, H, W, C)
    return _BM


def get_block_manager() -> BlockManager:
    global _BM
    assert _BM is not None, "BlockManager has not been initialized."
    return _BM
