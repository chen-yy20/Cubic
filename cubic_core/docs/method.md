# Cubic Methodology

本文档面向论文正文的 methodology 部分，而不是完整理论附录。写作目标是让读者顺着一条清晰的逻辑链读下来：

1. Flow matching 轨迹并非处处同样“直”；
2. 因此不同 token 对缓存复用的容忍度不同；
3. 我们先在线估计这种局部直线度，再据此设计缓存粒度；
4. 最后在给定计算预算下分配 block 更新间隔，并执行 selective token forward。

注意：latex版本中，将Cubic替换为/cubic。完整推导，包括局部截断误差界、Lipschitz/Grönwall 修补以及闭式优化解的细节，统一放在 appendix。正文只保留最关键的定义和结论。

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

长推导与与真实曲率的关系证明见 appendix。

---

## 3. Design Cache Granularity (Partitioner)

### 3.1 From Token Curvature to Cache Granularity

有了 token-space curvature map 之后，下一个问题不是立刻给每个 token 分配独立更新周期，而是先确定缓存应该采用多粗的空间粒度。原因是 selective execution 的基本单元不是单个 token，而是共享缓存策略的一组 token。

因此，Cubic 在 token space 上定义 block partition，但 partition 本身也是自适应的：在每个 anchor step，我们从候选 block sizes 中选择一个全局 block size，然后在整个 token grid 上均匀平铺。换言之，**自适应的是粒度，均匀的是切分**。

### 3.2 Selection Criterion

对任意候选 block size $\mathbf{b}=(b_f,b_h,b_w)$，记其诱导的均匀分区为 $\mathcal{P}(\mathbf{b})=\{B_j(\mathbf{b})\}$。我们用块内 curvature 的变异系数衡量该粒度下的内部一致性：

$$
\mathrm{cv}_j(\mathbf{b})
=
\frac{\mathrm{Std}(\{\kappa_i\}_{i \in B_j(\mathbf{b})})}
{\mathrm{Mean}(\{\kappa_i\}_{i \in B_j(\mathbf{b})}) + \varepsilon}.
$$

我们的选择原则很简单：优先选择**块内曲率更一致**、同时**块尽可能大**的划分。实现上，这一准则通过 block-wise CV 的统计量来近似，包括 mean-CV 与 p90-CV；若某个候选已满足一致性阈值，则优先采用体积最大的 block size，否则退化为选择 p90-CV 最小的候选。

### 3.3 Representative Block Curvature

在确定分区后，我们需要把 token-wise curvature 汇聚成 block-wise 调度信号。当前实现中，我们不使用 block 内均值，而使用更保守的高分位数：

$$
\kappa_b = \mathrm{Quantile}_{q}\big(\{\kappa_i\}_{i \in B_b}\big), \qquad q \approx 0.9.
$$

这样做的动机是避免大块中的少数高曲率 token 被均值“淹没”，从而降低过度跳步带来的局部失真风险。

---

## 4. Cache Error Modeling and Interval Optimization (Optimizer)

### 4.1 Error Model

一旦某个 block 在当前步完成刷新，我们在接下来若干步内复用其缓存输出，此处应该强调insight：等价于在局部时间窗口上对速度场应用 zero-order hold。为避免与前文的终点预测漂移 $\hat{\delta}_{i}^{(m)}$ 混淆，我们将 ZOH 引入的局部状态误差记为 $e_b$。对于 block $b$，若其曲率为 $\kappa_b$，跳过 $\tau_b$ 步带来的局部状态误差满足

$$
\|e_{\tau_b}^{(b)}\|
\;\lesssim\;
\frac{1}{2}\,\kappa_b\,\tau_b^2\,\Delta t^2.
$$

这个关系给出了 Cubic 的核心直觉：更直的 block 对长间隔缓存复用更鲁棒，更弯的 block 则需要更频繁地刷新。

### 4.2 Budget-Constrained Interval Allocation

接下来我们将 block 更新间隔的分配写成一个受预算约束的优化问题。令 $\rho$ 为目标加速比，$\tau_b$ 为 block $b$ 的更新间隔。我们将 block error parameterize 为

$$
\ell_b(\tau_b) \propto \kappa_b^\alpha \tau_b^\beta,
$$

并在平均更新率受限的条件下最小化总误差，即

$$
\min_{\tau_b \ge 1}\; \sum_{b=1}^{B} \kappa_b^\alpha \tau_b^\beta
\qquad
\text{s.t.}\quad
\frac{1}{B}\sum_{b=1}^{B}\frac{1}{\tau_b} = \frac{1}{\rho}.
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

在实现中，我们进一步加入离散化、上下界裁剪以及目标 speedup 的重新归一化，使得到的 $\tau_b$ 为整数并尽量接近用户指定的加速比。完整推导与离散修正细节见 appendix。

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
