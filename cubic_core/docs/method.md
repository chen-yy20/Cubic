# Cubic Methodology

本文档面向论文正文的 methodology 部分，而不是完整理论附录。写作目标是让读者顺着一条清晰的逻辑链读下来：

1. Flow matching 轨迹并非处处同样“直”；
2. 因此不同 token 对缓存复用的容忍度不同；
3. 我们先在线估计这种局部直线度，再据此设计缓存粒度；
4. 最后在给定计算预算下分配 block 更新间隔，并执行 selective token forward。

注意：LaTeX 版本中，将 Cubic 替换为 `\cubic`。完整推导，包括局部截断误差界、Lipschitz/Grönwall 修补以及闭式优化解的细节，统一放在 appendix。正文只保留最关键的定义和结论。

---

## 1. Overview

我们考虑 flow matching sampler 的欧拉离散化：

$$
x_{k+1} = x_k + \Delta t_k \, v_\theta(x_k, t_k).
$$

Rectified Flow 的核心特性是：若速度场沿轨迹几乎不变，则生成轨迹近似直线，单步外推和缓存复用都会更可靠。Cubic 的基本思想是利用这种**局部直线度的空间异质性**：对轨迹更直的区域更激进地复用缓存，对轨迹更弯的区域更频繁地刷新计算。

具体来说，Cubic 在每个 anchor step 执行四个连续操作：

1. 用当前 full forward 构造 token-wise curvature proxy；
2. 在 token space 上选择合适的 cache granularity；
3. 将 block curvature 映射为 block-specific update interval；
4. 在后续 selective steps 中仅重算 active tokens，并复用 frozen tokens 的缓存结果。

为了保证最终质量，我们将整个采样过程划分为 warmup, cruise, cooldown 三段：warmup 和 cooldown 使用 full compute，cruise 阶段交替执行 anchor step 与 selective step。

---

## 2. Curvature Proxy (Monitor)

### 2.1 Motivation

我们希望找到一个无需额外前向传播、但能反映局部轨迹弯曲程度的在线指标。关键观察是：如果某个 token 的轨迹是直的，那么“以当前速度场直接外推到终点”的预测在相邻两次观测之间应几乎不变；反之，若轨迹弯曲，则这一终点预测会持续漂移。

### 2.2 Endpoint Prediction Drift

对于 token 位置 $i$，定义当前观测时刻的终点预测为

$$
\hat{x}_{1,i}(t) = x_i(t) + (1 - t)\, v_{\theta,i}(x(t), t).
$$

需要强调的是，Cubic 的 monitor 并不在每个 sampler step 都更新，而只在 anchor steps 上更新。因此我们实际比较的是**相邻两次 anchor observation** 之间的终点预测漂移。记 anchor 序列为 $\{t_{a_m}\}_{m=1}^M$，则漂移定义为

$$
\hat{\delta}_{i}^{(m)} = \hat{x}_{1,i}(t_{a_m}) - \hat{x}_{1,i}(t_{a_{m-1}}).
$$

其与局部轨迹曲率满足如下近似关系：

$$
\left\|\hat{\delta}_{i}^{(m)}\right\|
\approx
\kappa_i(t_{a_m})\,(1-t_{a_m})\,\Delta t_{a_m},
$$

其中 $\Delta t_{a_m} \triangleq |t_{a_m} - t_{a_{m-1}}|$ 是相邻两次 anchor 之间的时间间隔。

因此，$\|\hat{\delta}_{i}^{(m)}\| / ((1-t_{a_m})\Delta t_{a_m})$ 可以作为 token-wise curvature 的在线 proxy。实现中，我们先在 latent space 上计算该 proxy，再按照与模型 patch size 一致的方式聚合到 token grid，得到 token-space curvature map

$$
\kappa \in \mathbb{R}^{F_t \times H_t \times W_t}.
$$

### 2.3 Why This Proxy Matters

这一步给出了 Cubic 后续所有决策的统一信号源。重要的是，monitor 并不直接决定哪个 token 被跳过；它提供的是一个 token-space risk map，后续的 partitioner 和 optimizer 在此基础上进一步决定缓存粒度与更新频率。

长推导以及与真实曲率的关系证明见 appendix。

---

## 3. Design Cache Granularity (Partitioner)

### 3.1 From Token Curvature to Morphology-guided Blocks

有了 token-space curvature map $\kappa \in \mathbb{R}^{F_t \times H_t \times W_t}$ 之后，下一个问题不是立刻给每个 token 分配独立更新周期，而是先确定缓存策略应该在哪些 token 组之间共享。原因是 selective execution 的实际调度单元是 block：同一 block 内 token 共享刷新间隔 $\tau_b$。

当前实现默认采用 **morphology-guided non-uniform partition**。直觉上，高曲率区域往往对应更需要精细更新的前景或局部细节，而低曲率区域可以使用更粗的缓存粒度。因此，Cubic 先将时间维平均后的空间曲率图

$$
\bar{\kappa}_{h,w} = \frac{1}{F_t}\sum_{f=1}^{F_t}\kappa_{f,h,w}
$$

转化为前景掩码，再基于该掩码生成非重叠的矩形 block。这样得到的分区不再要求全局统一 block size，而是在高曲率/混合区域自动细分，在低曲率背景区域保持较大 block。

### 3.2 Morphological Partitioning

具体地，Cubic 先根据 $\bar{\kappa}$ 的高分位阈值构造候选前景集合

$$
M_{h,w} = \mathbb{1}\!\left[\bar{\kappa}_{h,w} \ge
\mathrm{Quantile}_{q_f}(\bar{\kappa})\right],
$$

其中 $q_f$ 是前景阈值分位数。随后对 $M$ 执行轻量的形态学清理，包括 opening、closing、dilation/erosion 以及小连通域过滤，得到更稳定的前景掩码 $\tilde{M}$。这些操作只用于稳定分区边界，不引入额外模型前向。

在 $\tilde{M}$ 上，partitioner 从完整空间区域开始递归切分。若某个区域的前景比例处于混合区间、块内曲率 CV 超过阈值，或区域大小超过当前前景/背景目标粒度，则沿更优的空间轴继续二分；否则该区域成为一个 block。前景占比较高的区域使用更小目标粒度，背景占比较低的区域使用更大目标粒度。最终得到

$$
\mathcal{P}=\{B_b\}_{b=1}^{B}, \qquad
\bigcup_{b=1}^{B} B_b = \Omega,\quad B_b \cap B_{b'}=\varnothing.
$$

这个设计保留了规则矩形 block 带来的索引与调度便利，同时允许缓存粒度随局部曲率结构自适应变化。实现中仍保留 uniform partition 作为 fallback 或显式配置选项；当缺少有效 curvature map 或 morphology 分区失败时，会退化到均匀切分。

### 3.3 Representative Block Curvature

在确定分区后，我们需要把 token-wise curvature 汇聚成 block-wise 调度信号。当前通用实现使用 block 内均值：

$$
\kappa_b
=
\frac{1}{|B_b|}
\sum_{i \in B_b}\kappa_i .
$$

由于分区阶段已经倾向于把高曲率/混合区域切得更细，均值聚合可以作为稳定的 block-level 风险估计，并直接传递给后续 interval optimizer。

---

## 4. Cache Error Modeling and Interval Optimization (Optimizer)

### 4.1 Error Model

一旦某个 block 在当前步完成刷新，我们在接下来若干步内复用其缓存输出。这等价于在局部时间窗口上对速度场应用 zero-order hold。为避免与前文的终点预测漂移 $\hat{\delta}_{i}^{(m)}$ 混淆，我们将 ZOH 引入的局部状态误差记为 $e_b$。对于 block $b$，若其曲率为 $\kappa_b$，跳过 $\tau_b$ 步带来的局部状态误差满足

$$
\|e_{\tau_b}^{(b)}\|
\;\lesssim\;
\frac{1}{2}\,\kappa_b\,\tau_b^2\,\Delta t^2.
$$

这个关系给出了 Cubic 的核心直觉：更直的 block 对长间隔缓存复用更鲁棒，更弯的 block 则需要更频繁地刷新。

### 4.2 Budget-Constrained Interval Allocation

接下来我们将 block 更新间隔的分配写成一个受预算约束的优化问题。令 $\rho$ 为目标加速比，$\tau_b$ 为 block $b$ 的更新间隔，$n_b=|B_b|$ 为该 block 的 token 数。我们将 block error parameterize 为

$$
\ell_b(\tau_b) \propto \kappa_b^\alpha \tau_b^\beta,
$$

并在 token-weighted 平均更新率受限的条件下最小化总误差，即

$$
\min_{\tau_b \ge 1}\; \sum_{b=1}^{B} \kappa_b^\alpha \tau_b^\beta
\qquad
\text{s.t.}\quad
\frac{\sum_{b=1}^{B} n_b/\tau_b}{\sum_{b=1}^{B} n_b}
= \frac{1}{\rho}.
$$

其闭式最优解满足

$$
\tau_b^\star \propto \kappa_b^{-\alpha/(\beta+1)}.
$$

当采用理论默认值 $(\alpha,\beta)=(2,4)$ 时，可得到

$$
\tau_b^\star \propto \kappa_b^{-2/5}.
$$

这给出一个清晰结论：高曲率 block 应该更频繁刷新，低曲率 block 可以更长时间复用缓存。

在实现中，我们进一步加入目标 speedup 的重新归一化、离散化以及合法值下界处理，使得到的 $\tau_b$ 为正整数并尽量接近用户指定的加速比。完整推导与离散修正细节见 appendix。

---

## 5. Selective Token DiT Forward (Engine)

### 5.1 Active and Frozen Tokens

给定分区 $\mathcal{P}$ 和 block intervals $\{\tau_b\}$ 后，在 selective step $k$，active blocks 定义为

$$
\mathcal{A}_k = \{ b \in \mathcal{P} \mid s_k \bmod \tau_b = 0 \},
$$

其中 $s_k$ 是 selective phase 内的局部步编号。对应地，

$$
\Omega_k^{\mathrm{act}} = \bigcup_{b \in \mathcal{A}_k} B_b,
\qquad
\Omega_k^{\mathrm{frz}} = \Omega \setminus \Omega_k^{\mathrm{act}}.
$$

也就是说，Cubic 实际选择的不是“哪些 block 在理论上重要”，而是“当前这一步哪些 token 需要重新计算，哪些 token 可以安全复用缓存”。

### 5.2 Selective Transformer Execution

在 transformer 的每一层，Cubic 仅对 active tokens 重新计算 query/key/value 与 FFN；frozen tokens 则直接复用先前缓存的层内表示或最终输出。可将这一过程概括为：

$$
Y_\ell^{\mathrm{act}}
=
\mathrm{Attn}\!\left(
Q_\ell^{\mathrm{act}},
K_\ell^{\mathrm{full}},
V_\ell^{\mathrm{full}}
\right),
$$

其中 $K_\ell^{\mathrm{full}}$ 和 $V_\ell^{\mathrm{full}}$ 由 active tokens 的新 K/V 与 frozen tokens 的缓存 K/V 拼接而成。最终一步 selective forward 的输出可以写成

$$
v_k(\Omega)
=
\mathrm{Merge}\!\left(
v_k^{\mathrm{act}}(\Omega_k^{\mathrm{act}}),
v_{k-1,\mathrm{cache}}(\Omega_k^{\mathrm{frz}})
\right).
$$

从实现角度看，这一步才是真正带来 FLOPs 节省的部分：active tokens 走完整计算路径，frozen tokens 则只参与缓存复用而不再重复执行完整的 DiT block。

### 5.3 Runtime Stabilization

为了保证数值稳定性，Cubic 只在 cache 已经建立且 shape 合法时执行 selective forward；否则自动回退到 full forward 刷新缓存。这个 bootstrap/refresh 机制保证了 selective path 始终依附于一个有效的全量参考状态。

---

## 6. Putting Everything Together

综合以上四个模块，Cubic 的在线推理流程可以概括为：

1. 在 warmup 阶段执行 full forward，建立初始曲率观测与缓存状态。
2. 在每个 anchor step，使用 full forward 更新 token-space curvature proxy，并重新选择 block granularity 与 block intervals。
3. 在相邻的 selective steps 中，仅对 active tokens 执行 selective DiT forward，复用 frozen tokens 的缓存结果。
4. 在 cooldown 阶段恢复 full forward，以保证最终生成质量。

这一设计将“局部轨迹几何”与“缓存执行策略”直接连接起来：monitor 负责观测曲率，partitioner 负责设计粒度，optimizer 负责分配更新间隔，engine 负责将这些决策落实为实际的 selective token computation。
