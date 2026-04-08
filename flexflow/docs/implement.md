# Cubic 推理加速系统 — 实现指南

> 本文档用于指导 Agent 实现 Cubic 系统的完整代码。
> 请严格遵循本文档中的接口规格、数据流设计和约束条件。

> Jano: Adpative Diffusion Generation with Early-stage Convergence Awareness 是Cubic的前置工作，可以参考`../reference/`中的Jano代码实现来实现cubic的相关代码。

---

## 1. 项目概述

### 1.1 目标

Cubic 是一个面向 Flow-Matching DiT（Diffusion Transformer）的推理加速框架。其核心思想是：在 ODE 采样过程中，速度场的不同空间区域具有不同的"直度"（straightness），直度高的区域可以跳步复用缓存速度，直度低的区域需要每步重新计算。通过自适应监测与分块调度，Cubic 在保持生成质量的前提下减少总计算量。

### 1.2 加速机制

真正的 FLOPs 节省来自**稀疏 Token 计算**：在 DiT 的 Transformer 层中，仅对 active block 的 token 计算 Q/K/V 投影和 FFN，frozen block 的 token 通过 per-layer KV cache 参与 attention 的 K/V 端，但不计算 Q 也不过 FFN。

### 1.3 前置假设

- 模型为标准 DiT 架构：patch embedding → L 个 transformer block → unpatchify
- 每个 transformer block 包含 self-attention + FFN（可能有 cross-attention）
- 采样器为 Euler method：`x_{k+1} = x_k + dt * v_theta(x_k, t_k)`
- 时间从 `t=0`（噪声）到 `t=1`（数据），共 K 步
- 输入 x 形状为 `(B, C, F, H, W)`，其中 F 为帧数，H/W 为空间尺寸

---

## 2. 核心算法

### 2.1 三阶段推理

整个 K 步采样分为三个阶段：

| 阶段 | 步数范围 | 行为 |
|------|---------|------|
| Warmup | `[0, K_w)` | 每步 full forward，用于建立初始曲率估计 |
| Cruise | `[K_w, K - K_c)` | 选择性稀疏计算，是加速的核心阶段 |
| Cooldown | `[K - K_c, K)` | 每步 full forward，保证最终质量 |

其中 `K_w = floor(K * warmup_fraction)`，`K_c = floor(K * cooldown_fraction)`。

### 2.2 Anchor Step

在 cruise 阶段，每隔 `anchor_interval` 步设置一个 anchor step。Anchor step 执行：

1. Full forward（全量计算）
2. 计算终点预测并更新曲率图
3. 基于曲率图执行自适应重分块
4. 为新分块分配更新频率 τ_b
5. 刷新所有缓存（KV cache + velocity cache）

非 anchor 的 cruise step 执行稀疏 forward：仅计算 active block 的 token。

### 2.3 终点预测曲率

在 anchor step t_k，利用 full forward 得到的 v，计算终点预测：$$x̂_1(t_k) = x(t_k) + (1 - t_k) * v(t_k)$$


两个相邻 anchor 之间的终点漂移：

$$δ = x̂_1(t_{k+M}) - x̂_1(t_k)$$

逐 token 曲率（在 latent 空间，沿 C 维取范数）：

$$κ_i = ||δ_i||2 / ((1 - t{k+M}) * |t_{k+M} - t_k| + eps)$$

`κ_i` 越大，该位置的 ODE 轨迹越弯曲，越需要频繁更新。

### 2.4 自适应四叉树分块

输入：曲率图 `κ ∈ R^{Ft × Ht × Wt}`（token 空间）  
输出：一组不重叠、完全覆盖的块 `{Block_b}`

算法：对每一帧（或帧组），在 (Ht, Wt) 平面上做四叉树递归分裂。分裂条件：

```
cv = std(κ[region]) / (mean(κ[region]) + eps)
split if cv > cv_threshold AND region_size > min_block_size
```

约束：所有块边长必须是 2 的幂次，且 ≥ min_block_size、≤ max_block_size。

### 2.5 τ_b 优化分配

每个 block b 的代表曲率 `κ_b = percentile_90(κ[block_b])`。

连续最优解（误差-计算量权衡）：

```
τ_b ∝ κ_b^{-α/(β+1)}
```
默认 α=2, β=4，所以指数为 -2/5。

归一化约束：所有块的平均更新率 `mean(1/τ_b)` 等于 `1/target_speedup`。

最终将连续解离散化（四舍五入 + clip 到 [τ_min, τ_max]）。

### 2.6 稀疏 Attention 【参考`jano_maskcache.py`和`wan_modules.attention.py`】

在 cruise 的非 anchor step：

对于 DiT 的第 l 层 transformer block：
- **Active tokens**（属于需要更新的 block）：正常计算 Q_l, K_l, V_l 投影，正常过 FFN
- **Frozen tokens**（属于跳步复用的 block）：不计算任何东西，使用 per-layer KV cache 中存储的 K_l, V_l
- **Attention 计算**：`Attention(Q_active, K_full, V_full)`，其中 `K_full = scatter(K_active, K_cached_frozen)` 

frozen token 的最终输出速度直接从 velocity cache 中取。

---

## 3. 代码架构

### 3.1 文件结构

```
cubic/
├── init.py # 导出 CubicScheduler, CubicConfig
├── config.py # 超参数配置
├── types.py # 核心数据结构 Block, Partition, Phase
├── partition.py # 自适应四叉树分块
├── monitor.py # 曲率监测
├── optimizer.py # τ_b 优化分配
├── cache.py # KV cache + velocity cache 管理
├── transformer.py - selective_forward.py # 计算部分token的dit forward
├── scheduler.py # 主调度器（唯一对外接口）
└── utils.py # 工具函数
```

---

`scheduler.py` 是唯一的协调者。其他模块之间无直接依赖（除了共享 `types.py`）。

### 3.3 对外接口

用户只需与 `CubicScheduler` 交互：

```python
from cubic import CubicScheduler, CubicConfig

config = CubicConfig(target_speedup=2.0)
scheduler = CubicScheduler(config=config)

# 初始化：传入模型和 attention layer 的引用
scheduler.setup(model, attention_layers_or_hooks)

# 推理循环
x = torch.randn(B, C, F, H, W)
for k, t in enumerate(timesteps):
    # 适配了cubic的model forward

```

## 4.模块详细规格

### 4.1 `config.py`

``` python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CubicConfig:
    """Cubic 系统的所有超参数。"""

    # --- 加速目标 ---
    target_speedup: float = 2.0          # ρ，平均每 ρ 步才做一次 full compute

    # --- 分块参数 ---
    min_block_size: int = 2              # 四叉树最小块边长（token 单位）
    max_block_size: int = 16             # 四叉树最大块边长（token 单位）
    cv_threshold: float = 0.5            # 变异系数阈值，越小分得越细

    # --- 曲率 → τ 映射 ---
    alpha: float = 2.0                   # 误差阶次
    beta: float = 4.0                    # 计算量阶次
    tau_min: int = 1                     # 最小跳步间隔（1=每步更新）
    tau_max: int = 10                    # 最大跳步间隔

    # --- 推理阶段 ---
    warmup_fraction: float = 0.15        # warmup 占总步数的比例
    cooldown_fraction: float = 0.15      # cooldown 占总步数的比例
    anchor_interval: int = 8             # cruise 阶段的 anchor 间隔

    # --- 模型相关 ---
    patch_size: tuple = (1, 2, 2)        # DiT 的 patch size (pt, ph, pw)
    num_layers: int = 30                 # Transformer 层数（用于初始化 KV cache）

    # --- 曲率聚合 ---
    curvature_percentile: float = 0.9    # block 内曲率聚合的分位数
```

---

### 4.2 `types.py`

``` python
"""核心数据结构定义。"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import torch

class Phase(Enum):
    WARMUP = "warmup"
    CRUISE_ANCHOR = "cruise_anchor"    # cruise 阶段中的 anchor step
    CRUISE_SELECTIVE = "cruise_selective"    # cruise 阶段中的普通 step
    COOLDOWN = "cooldown"


@dataclass
class Block:
    """token 空间中的一个空间块。
    
    坐标系：(Ft, Ht, Wt) = (F/pt, H/ph, W/pw) 的 token grid。
    block 定义为该 grid 上的一个矩形区域。
    """
    block_id: int
    f_start: int
    f_end: int
    h_start: int
    h_end: int
    w_start: int
    w_end: int

    @property
    def token_count(self) -> int:
        return (self.f_end - self.f_start) * \
               (self.h_end - self.h_start) * \
               (self.w_end - self.w_start)

    def get_token_indices(self, Ft: int, Ht: int, Wt: int) -> torch.Tensor:
        """返回该 block 内所有 token 在 flatten 后序列中的索引。
        
        token 的 flatten 顺序为 (f, h, w) → f * Ht * Wt + h * Wt + w。
        
        返回: LongTensor of shape (token_count,)
        """
        indices = []
        for f in range(self.f_start, self.f_end):
            for h in range(self.h_start, self.h_end):
                for w in range(self.w_start, self.w_end):
                    indices.append(f * Ht * Wt + h * Wt + w)
        return torch.tensor(indices, dtype=torch.long)
    
    # 注意：上面的循环实现仅用于明确语义。
    # 实际实现请用向量化方式：
    # f_idx = torch.arange(f_start, f_end)
    # h_idx = torch.arange(h_start, h_end)
    # w_idx = torch.arange(w_start, w_end)
    # grid = torch.stack(torch.meshgrid(f_idx, h_idx, w_idx, indexing='ij'), dim=-1)
    # indices = grid[...,0]*Ht*Wt + grid[...,1]*Wt + grid[...,2]
    # return indices.reshape(-1)


@dataclass
class Partition:
    """一个完整的分块方案。所有 block 不重叠且完全覆盖整个 token grid。"""
    blocks: List[Block]
    Ft: int     # token grid 的时间维大小
    Ht: int     # token grid 的高度
    Wt: int     # token grid 的宽度

    def __len__(self):
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks)

    def get_all_token_indices(self) -> Dict[int, torch.Tensor]:
        """返回 {block_id: token_indices} 的映射。"""
        return {
            b.block_id: b.get_token_indices(self.Ft, self.Ht, self.Wt)
            for b in self.blocks
        }

    @property
    def total_tokens(self) -> int:
        return self.Ft * self.Ht * self.Wt


@dataclass
class StepPlan:
    """单步推理计划，由 scheduler 生成，传递给 selective_forward。"""
    phase: Phase
    active_block_ids: List[int]           # 本步需要更新的 block id 列表
    active_token_indices: torch.Tensor    # 所有 active block 的 token 索引（已合并、排序）
    frozen_token_indices: torch.Tensor    # 所有 frozen block 的 token 索引
    is_full_compute: bool                 # True if phase != CRUISE_selective
```

---

### 4.3 `partition.py` 【保守参考`jano_blockmanager.py`】

```python
"""自适应四叉树分块。"""

import torch
from typing import List, Optional
from .types import Block, Partition
from .config import CubicConfig


class AdaptivePartitioner:
    """基于曲率图的自适应空间分块。"""

    def __init__(self, config: CubicConfig):
        self.min_block_size = config.min_block_size
        self.max_block_size = config.max_block_size
        self.cv_threshold = config.cv_threshold

    def partition(
        self,
        Ft: int,
        Ht: int,
        Wt: int,
        curvature_map: Optional[torch.Tensor] = None,
    ) -> Partition:
        """生成分块方案。
        
        Args:
            Ft, Ht, Wt: token grid 尺寸
            curvature_map: (Ft, Ht, Wt) 曲率图。若为 None，返回均匀分块。
        
        Returns:
            Partition 对象
        """
        # 实现要点：
        # 1. 若 curvature_map 为 None，调用 _uniform_partition
        # 2. 否则，对 Ft 维度按 min_block_size 分段（或不分，每帧独立）
        #    对每段在 (Ht, Wt) 平面上执行四叉树分裂
        # 3. 返回 Partition

    def _uniform_partition(self, Ft, Ht, Wt) -> Partition:
        """均匀分块，块大小为 max_block_size。"""
        # 实现：三重循环，步长为 max_block_size（或 min(max_block_size, dim)）
        # 注意处理不整除的边界

    def _quadtree_split(
        self,
        curvature_map: torch.Tensor,   # (Ft, Ht, Wt)
        f_start: int, f_end: int,
        h_start: int, h_end: int,
        w_start: int, w_end: int,
        blocks: List[Block],
        counter: List[int],            # 用 list 包装以支持闭包内修改
    ):
        """递归四叉树分裂。
        
        分裂判据：
            region = curvature_map[f_start:f_end, h_start:h_end, w_start:w_end]
            cv = region.std() / (region.mean() + 1e-8)
            split = (cv > self.cv_threshold) 
                    and (h_size > self.min_block_size) 
                    and (w_size > self.min_block_size)
        
        若不分裂（或已达最小尺寸），创建 Block 并 append 到 blocks。
        若分裂，将 H/W 各一分为二，递归处理 4 个子区域。
        
        注意：分裂点取 h_mid = h_start + h_size // 2（保证是 2 的幂对齐，
        前提是初始尺寸是 2 的幂。若不是，向下取最近的偶数。）
        """
```

实现约束：

* 所有 block 的 f/h/w 范围必须恰好覆盖 [0, Ft) × [0, Ht) × [0, Wt)，无遗漏无重叠
* block_id 从 0 递增，唯一
* 在 F 维度上可以简化处理：若 Ft 较小（如 ≤4），F 维度不分裂，整体作为一层

---

### 4.4 `monitor.py` [保守参考`jano_analyzer.py`]

```python
"""曲率监测模块。"""

import torch
from typing import Optional, Dict
from .types import Partition
from .config import CubicConfig


class CurvatureMonitor:
    """追踪终点预测漂移，计算逐 token 曲率。"""

    def __init__(self, config: CubicConfig):
        self.config = config
        self.prev_endpoint: Optional[torch.Tensor] = None   # (B, C, F, H, W)
        self.prev_t: Optional[float] = None
        self.curvature_map: Optional[torch.Tensor] = None   # (Ft, Ht, Wt) token space

    def update(
        self,
        x: torch.Tensor,      # (B, C, F, H, W) 当前状态
        v: torch.Tensor,       # (B, C, F, H, W) 当前速度
        t: float,              # 当前时间
    ) -> Optional[torch.Tensor]:
        """在 anchor step 调用。
        
        计算终点预测并与上次 anchor 比较，更新曲率图。
        
        Returns:
            curvature_map: (Ft, Ht, Wt) 的逐 token 曲率，首次调用返回 None。
        """
        # 步骤：
        # 1. 计算终点预测: endpoint = x + (1 - t) * v  → (B, C, F, H, W)
        # 2. 若 self.prev_endpoint 存在：
        #    a. drift = endpoint - self.prev_endpoint     → (B, C, F, H, W)
        #    b. drift_norm = drift.norm(dim=1)            → (B, F, H, W)  沿 C 维
        #    c. normalizer = (1 - t) * |t - self.prev_t| + eps
        #    d. curvature_latent = drift_norm / normalizer → (B, F, H, W)
        #    e. 对 batch 维求均值                          → (F, H, W)
        #    f. 下采样到 token 空间                        → (Ft, Ht, Wt)
        #       使用 avg_pool3d 或 reshape + mean，pool 核为 patch_size
        #    g. self.curvature_map = 结果
        # 3. 更新 self.prev_endpoint, self.prev_t
        # 4. 返回 self.curvature_map（可能为 None）

    def get_block_curvatures(self, partition: Partition) -> Dict[int, float]:
        """将逐 token 曲率聚合为逐 block 曲率。
        
        每个 block 的代表曲率 = 其区域内曲率的 90th percentile。
        
        Returns:
            {block_id: κ_b}
        """
        # 注意：self.curvature_map 为 None 时应返回空 dict 或 raise
        # 对每个 block，提取 curvature_map[f_start:f_end, h_start:h_end, w_start:w_end]
        # 计算 torch.quantile(region.float().flatten(), self.config.curvature_percentile)

    def reset(self):
        """清除所有状态。在推理开始时调用。"""
        self.prev_endpoint = None
        self.prev_t = None
        self.curvature_map = None
```

---


### 4.5 `optimizer.py`

```python
"""τ_b 优化分配。"""

from typing import Dict
from .config import CubicConfig


class ScheduleOptimizer:
    """根据曲率和加速目标，计算每个 block 的最优更新间隔 τ_b。"""

    def __init__(self, config: CubicConfig):
        self.config = config
        self.exponent = config.alpha / (config.beta + 1.0)  # 默认 2/5 = 0.4

    def optimize(
        self,
        block_curvatures: Dict[int, float],
    ) -> Dict[int, int]:
        """
        Args:
            block_curvatures: {block_id: κ_b}
        
        Returns:
            {block_id: τ_b}，τ_b ∈ [τ_min, τ_max] 的整数
        """
        # 算法：
        # 1. raw_b = κ_b^{-exponent}  对每个 block
        # 2. 求缩放系数 c 使得 mean(1 / (c * raw_b)) = 1 / target_speedup
        #    即 c = target_speedup * mean(1 / raw_b)
        #    推导：
        #      mean(1/τ_b) = 1/ρ
        #      τ_b = c * raw_b
        #      mean(1/(c*raw_b)) = 1/ρ
        #      (1/c) * mean(1/raw_b) = 1/ρ
        #      c = ρ * mean(1/raw_b)
        # 3. τ_b = round(clip(c * raw_b, τ_min, τ_max))
        # 4. 确保所有 τ_b ≥ 1
        
        # 边界情况：若所有 κ_b ≈ 0（极直轨迹），raw_b → ∞，
        # clip 到 τ_max 即可。
        # 若某些 κ_b 极大（极弯曲），raw_b → 0，clip 到 τ_min=1。
```

---

### 4.6 `cache.py` 【参考jano_maskcache.py】

要点：
1. frozen KV可以直接拼接，而不需要restore到原位。
2. model output tensor也需要被储存。
3. offload功能可以暂时先不实现。


---

### 4.7 `scheduler.py` 

```python
"""主调度器。"""

import torch
from typing import Optional
from .config import CubicConfig
from .types import Phase, StepPlan, Partition
from .monitor import CurvatureMonitor
from .blocksplit import AdaptivePartitioner
from .optimizer import ScheduleOptimizer
from .cache import VelocityCache
from .selective_forward import CubicModelWrapper


class CubicScheduler:
    """Cubic 推理加速主调度器。
    
    职责：
    1. 判断每步的 phase
    2. 在 anchor step 触发监测→分块→调度更新
    3. 生成 StepPlan
    4. 调用 CubicModelWrapper 执行 forward
    5. 管理 velocity cache 的 tick / update
    """

    def __init__(self, config: CubicConfig):
        self.config = config
        self.monitor = CurvatureMonitor(config)
        self.partitioner = AdaptivePartitioner(config)
        self.optimizer = ScheduleOptimizer(config)
        
        self.wrapper: Optional[CubicModelWrapper] = None
        self.partition: Optional[Partition] = None
        self.Ft: int = 0
        self.Ht: int = 0
        self.Wt: int = 0
        self.initialized = False

    def setup(self, model, **model_specific_kwargs):
        """初始化，传入模型。
        
        Args:
            model: 原始 DiT 模型
            **model_specific_kwargs: 传递给 CubicModelWrapper.prepare() 的参数
        """
        self.wrapper = CubicModelWrapper(model, self.config)
        self.wrapper.prepare(**model_specific_kwargs)

    def step(
        self,
        x: torch.Tensor,       # (B, C, F, H, W)
        t: float,
        dt: float,
        step_idx: int,
        total_steps: int,
        **model_kwargs,         # 传递给模型的额外参数（text_embeds 等）
    ) -> torch.Tensor:
        """执行一步推理。
        
        Returns:
            v: (B, C, F, H, W) 速度场
        """
        # --- 延迟初始化 ---
        if not self.initialized:
            self._lazy_init(x)

        # --- 判断 phase ---
        phase = self._get_phase(step_idx, total_steps)

        # --- 生成 step plan ---
        step_plan = self._make_plan(phase)

        # --- 执行 forward ---
        # 【参考jano的transformer forward实现】

        # --- 后处理 ---
        self._post_step(x, v, t, dt, step_idx, total_steps, phase, step_plan)

        return v

    def _lazy_init(self, x: torch.Tensor):
        """首次调用时，根据输入形状初始化分块和缓存。"""
        B, C, F, H, W = x.shape
        pt, ph, pw = self.config.patch_size
        self.Ft = F // pt
        self.Ht = H // ph
        self.Wt = W // pw

        # 初始均匀分块
        self.partition = self.partitioner.partition(self.Ft, self.Ht, self.Wt)
        self.wrapper.velocity_cache.reset_ages(self.partition)

        # 初始 schedule：所有块 τ=1（首步全部更新）
        initial_schedule = {b.block_id: 1 for b in self.partition}
        self.wrapper.velocity_cache.update_schedule(initial_schedule)

        self.initialized = True

    def _get_phase(self, step_idx: int, total_steps: int) -> Phase:
        """判断当前步属于哪个阶段。"""
        K_w = int(total_steps * self.config.warmup_fraction)
        K_c = int(total_steps * self.config.cooldown_fraction)

        if step_idx < K_w:
            return Phase.WARMUP
        elif step_idx >= total_steps - K_c:
            return Phase.COOLDOWN
        else:
            # Cruise 阶段，判断是否为 anchor
            cruise_offset = step_idx - K_w
            if cruise_offset % self.config.anchor_interval == 0:
                return Phase.CRUISE_ANCHOR
            else:
                return Phase.CRUISE_SELECTIVE

    def _make_plan(self, phase: Phase) -> StepPlan:
        """生成当前步的执行计划。"""
        all_indices = torch.arange(self.partition.total_tokens)

        if phase in (Phase.WARMUP, Phase.COOLDOWN, Phase.CRUISE_ANCHOR):
            return StepPlan(
                phase=phase,
                active_block_ids=[b.block_id for b in self.partition],
                active_token_indices=all_indices,
                frozen_token_indices=torch.tensor([], dtype=torch.long),
                is_full_compute=True,
            )
        else:
            # CRUISE_SELECTIVE
            active_bids = self.wrapper.velocity_cache.get_active_block_ids(self.partition)
            token_index_map = self.partition.get_all_token_indices()

            active_indices = torch.cat([token_index_map[bid] for bid in active_bids]) \
                             if active_bids else torch.tensor([], dtype=torch.long)
            active_set = set(active_indices.tolist())
            frozen_indices = torch.tensor(
                [i for i in range(self.partition.total_tokens) if i not in active_set],
                dtype=torch.long
            )

            return StepPlan(
                phase=phase,
                active_block_ids=active_bids,
                active_token_indices=active_indices.sort().values,
                frozen_token_indices=frozen_indices.sort().values,
                is_full_compute=False,
            )

    def _post_step(self, x, v, t, dt, step_idx, total_steps, phase, step_plan):
        """步后处理：更新监测、调度、缓存。"""

        # 1. 更新 velocity cache
        if step_plan.is_full_compute:
            self.wrapper.velocity_cache.store_full(v)
            self.wrapper.velocity_cache.mark_updated(
                [b.block_id for b in self.partition]
            )
        else:
            # 只更新active token对应的kv
            # 或者直接取完整kv替换，避免细碎的访存

        # 2. Anchor 逻辑
        is_anchor = phase in (Phase.CRUISE_ANCHOR,)
        is_last_warmup = (phase == Phase.WARMUP and 
                          step_idx == int(total_steps * self.config.warmup_fraction) - 1)

        if is_anchor or is_last_warmup:
            curvature_map = self.monitor.update(x, v, t)
            if curvature_map is not None:
                # 重新分块
                new_partition = self.partitioner.partition(
                    self.Ft, self.Ht, self.Wt, curvature_map
                )
                self.partition = new_partition
                self.wrapper.velocity_cache.reset_ages(new_partition)

                # 重新分配 τ_b
                block_curvatures = self.monitor.get_block_curvatures(new_partition)
                schedule = self.optimizer.optimize(block_curvatures)
                self.wrapper.velocity_cache.update_schedule(schedule)

                # 由于本步是 full compute，缓存已经新鲜，标记所有块
                self.wrapper.velocity_cache.mark_updated(
                    [b.block_id for b in new_partition]
                )
        elif phase == Phase.WARMUP:
            # warmup 阶段的非末尾步也记录 endpoint（为 monitor 提供首个 anchor 数据）
            # 仅在倒数第二步开始记录即可，或者只在 is_last_warmup 时记录
            # 为简化，warmup 阶段不记录 endpoint，仅在最后一步记录
            pass

        # 3. Tick
        self.wrapper.velocity_cache.tick()
```


---

## 5. 推理流程完整走查
以 K=50 步、warmup_fraction=0.15、cooldown_fraction=0.15、anchor_interval=8 为例：

```
K_w = 7, K_c = 7, cruise = steps 7..42

Step 0-5:   WARMUP, full forward, 无 monitor 更新
Step 6:     WARMUP (最后一步之前), monitor.update() → 存储 prev_endpoint  
            （或者 step 5 和 step 6 两步各调用一次 monitor.update）
Step 7:     CRUISE_ANCHOR (cruise_offset=0), full forward
            monitor.update() → 得到 curvature_map → 分块 → τ 分配
            此后 partition 和 schedule 生效
Step 8-14:  CRUISE_SELECTIVE, selective forward
            每步根据 age 和 τ 判断哪些 block active
Step 15:    CRUISE_ANCHOR (cruise_offset=8), full forward
            重新 monitor → 重新分块 → 重新分配 τ
Step 16-22: CRUISE_SELECTIVE
...
Step 43-49: COOLDOWN, full forward
```

### 5.1 单步CRUISE_SELECTIVE的详细数据流

```reasonml
输入: x (B,C,F,H,W), t, dt

1. scheduler._get_phase() → CRUISE_SELECTIVE

2. scheduler._make_plan():
   - velocity_cache.get_active_block_ids(partition) 
     → 例如 blocks [0, 3, 7] 需要更新
   - 收集这些 block 的 token indices → active_token_indices
   - 其余为 frozen_token_indices
   → StepPlan

3. wrapper.selective_forward(x, t, step_plan):
   a. patch_embed(x) + pos_embed → tokens_all (B, N, D)
   b. tokens_active = tokens_all[:, active_idx]  (B, N_a, D)
   c. 对每层 l:
      - adaLN + QKV proj (仅 active) → q, k_new, v_new
      - apply_rope(q, positions[active_idx])
      - apply_rope(k_new, positions[active_idx])  
      - kv_cache.update_layer(l, k_new, v_new, active_idx)
      - k_full, v_full = kv_cache.get_full_kv(l)
      - attn = SDPA(q, k_full, v_full) → (B, N_a, H, D)
      - out_proj → (B, N_a, D)
      - residual + FFN (仅 active)
   d. final_layer(tokens_active)
   e. 组装完整速度:
      v = velocity_cache.cached_velocity.clone()
      将 tokens_active unpatchify 后写入 v 的 active 区域
   f. 返回 v

4. scheduler._post_step():
   - store_full(v), mark_updated(active_bids), tick()
   
5. 外部循环: FM sampler => x = x + dt * v
```


## 6 Edge Case
* 单步推理：若 total_steps=1，则只有一步，视为 WARMUP。
* 步数太少：若 total_steps < 5，warmup/cooldown 可能重叠。此时应回退到全量计算。
* 所有 block 都 active：这等价于 full forward，但走了 selective 路径。应在 _make_plan 中检测此情况并标记 is_full_compute=True。
* 空 active 列表：理论上不应发生（至少有一个 block 的 age ≥ τ）。但若初始化有误，应 fallback 到 full forward。