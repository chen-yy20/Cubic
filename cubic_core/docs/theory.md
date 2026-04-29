# Cubic: Curvature-driven Block-wise Inference Control

## 完整理论推导与思维链指导书 (Revised)

---

## §1 Flow Matching 基础

在 Rectified Flow / Flow Matching (FM) 框架下，生成模型学习一个速度场 $v_\theta(x_t, t)$，驱动样本沿从噪声 $x_0$ 到数据 $x_1$ 的轨迹演化：

$$
x_t = (1 - t)\, x_0 + t\, x_1, \quad t \in [0, 1] \tag{1.1}
$$

推理时，给定离散时间网格 $0 = t_0 < t_1 < \cdots < t_K = 1$，步长记为 $\Delta t = t_k - t_{k-1}$（均匀或非均匀均可），通过前向欧拉迭代：

$$
x_{k+1} = x_k + \Delta t \cdot v_\theta(x_k, t_k) \tag{1.2}
$$

Rectified Flow 的核心动机在于：训练目标鼓励学到的速度场 $v_\theta$ 沿时间方向尽可能恒定，即生成轨迹尽可能趋于**直线**。当轨迹为完美直线时，$dv_\theta / dt \equiv 0$，单步欧拉即可精确求解 ODE；反之，轨迹越弯曲，所需的离散步数越多。

---

## §2 经验观察：空间异质性

FM sampler 的骨干网络通常为 DiT，其内部张量形状为

$$
\mathbf{x} \in \mathbb{R}^{C \times F \times H \times W}
$$

（对于图像 $F=1$）。在实际实现中，DiT 本身在 token space 上工作。记其 token 网格为

$$
\Omega \triangleq \{(f,h,w)\mid 0 \le f < F_t,\; 0 \le h < H_t,\; 0 \le w < W_t\}
$$

其中 $(F_t, H_t, W_t)$ 为 token-space 尺寸。Cubic 的 monitor 先在 latent space 上估计曲率，再按照与模型 patch size 一致的方式把曲率聚合到该 token 网格上。随后，Cubic **并不是直接固定一套手工 block**，而是在每个 anchor step 基于当前 token-space 曲率图，从候选 block size 集合中选择一个**全局 block size**

$$
\mathbf{b} = (b_f, b_h, b_w)
$$

并用该尺寸对整个 token 网格做均匀平铺；边界块允许被截断。因此，Cubic 的 block 逻辑是：

- **自适应的是 block 粒度（global block size）**
- **均匀的是在给定粒度下的空间切分（uniform tiling）**
- **调度的是切分后各 block 的更新频率 $\tau_b$**

**经验观察 (Empirical Observation)：** 在固定总步数 $K$ 的采样过程中，不同块的生成轨迹表现出显著的**异质性 (Spatial Heterogeneity)**——

- 纹理平滑、变化缓慢的区域（如背景、大面积色块），其速度场沿时间方向几乎恒定，即轨迹近似直线，仅需很少的步数即可达到视觉收敛；
- 纹理复杂、高频细节密集的区域（如人脸、文字、主体边缘），其速度场持续调整方向，即轨迹弯曲程度大，需要更多步数才能忠实还原。

这一现象的本质并非"某些区域提前收敛"，而是**不同空间区域的生成轨迹具有不同的直线度 (Straightness)**。直线度高的区域对大步长（粗粒度离散化）的容忍度高；直线度低的区域则要求小步长以跟踪弯曲的轨迹。

**核心思路：**

> **观察** → 内容复杂度与收敛速度强相关
> **分析** → 底层原因是局部流直线度 (Local Flow Straightness) 决定了步长容忍度 (Step-size Tolerance)
> **设计** → 基于 token-space curvature proxy 先选择合适的 cache granularity（block size），再对块分配不同的更新频率（ZOH 缓存策略）

---

## §3 局部轨迹曲率：定义与分解

### §3.1 轨迹曲率的定义

在进入 partition 之前，我们先在 **token level** 定义局部轨迹曲率。对于 token 位置 $i \in \Omega$，其在真实 ODE 轨迹上的速度场随时间的**全导数**为：

$$
\frac{d v_{\theta,i}}{dt}(t) = \frac{\partial v_{\theta,i}}{\partial t} + \left(\nabla_{x_i} v_{\theta,i}\right) \cdot v_{\theta,i} \tag{3.1}
$$

其中右侧第一项是速度场的**显式时间变化率 (Explicit temporal rate)**，第二项是**对流传输项 (Advective transport)**，度量了因状态 $x_i$ 沿轨迹移动而引起的速度场隐式变化。

我们将 token $i$ 在时刻 $t$ 的**局部轨迹曲率 (Local Trajectory Curvature)** 定义为该全导数的范数：

$$
\kappa_i(t) \triangleq \left\| \frac{d v_{\theta,i}}{dt}(t) \right\| \tag{3.2}
$$

**直观含义：**

- $\kappa_i(t) = 0$ ⟹ 速度场在 token $i$ 上恒定，轨迹为直线，**完美直线度**
- $\kappa_i(t)$ 大 ⟹ 速度场在 token $i$ 上剧烈变化，轨迹弯曲，**直线度低**

这与 Rectified Flow 的 "straightness" 概念完全一致：Rectified Flow 的 reflow 操作旨在全局降低 $\kappa$，而我们的方法则利用 $\kappa$ 的**空间异质性**进行局部自适应调度。

### §3.2 空间局部性假设 (Spatial Locality Assumption)

在 DiT 等具有全局自注意力的架构中，token $i$ 的速度场严格来说依赖于所有 token 的状态。因此全导数应展开为：

$$
\frac{d v_{\theta,i}}{dt} = \frac{\partial v_{\theta,i}}{\partial t} + \sum_{j \in \Omega} \frac{\partial v_{\theta,i}}{\partial x_j} \cdot \frac{d x_j}{dt} \tag{3.3}
$$

为使理论在块级别闭环，我们引入空间局部性假设：

> **假设 1 (Spatial Locality)：** 在时间步 $\Delta t$ 内，跨 token 雅可比项 $\partial v_{\theta,i} / \partial x_j$（$j \neq i$）相对于自项可忽略，从而
>
> $$\frac{d v_{\theta,i}}{dt} \approx \frac{\partial v_{\theta,i}}{\partial t} + \left(\nabla_{x_i} v_{\theta,i}\right) \cdot v_{\theta,i}$$

该假设的合理性依据：

- **(a)** 多项研究表明 DiT 的 self-attention score 表现出较强的空间局部性和稀疏性，远距离 token 间的注意力权重衰减迅速。
- **(b)** 在短时间窗口 $\Delta t$ 内，跨 token 状态变化为 $O(\Delta t)$ 量级，其对当前位置速度场的扰动为二阶小量 $O(\Delta t^2)$，在一阶分析中自然消去。
- **(c)** 实验验证：将跨位置雅可比冻结与否对比，误差差异在统计意义上不显著。

> **注：** 严格而言，式 (3.3) 中的完整全导数可以通过向量-雅可比积 (VJP) 直接计算，不存在计算上的障碍。空间局部性假设的目的是简化理论分析，使 $\kappa_i(t)$ 成为仅依赖 token $i$ 本身状态的局部量；在后续的 `§4.3-§4.4` 中，我们再将 token-level 曲率聚合为 block-level 调度量。

### §3.3 利用三角不等式建立可计算上界

对式 (3.2) 应用三角不等式：

$$
\kappa_i(t) \le \left\|\frac{\partial v_{\theta,i}}{\partial t}\right\| + \left\|\nabla_{x_i} v_{\theta,i}\right\| \cdot \left\|v_{\theta,i}\right\| \tag{3.4}
$$

右侧的三个量均可在推理过程中估计：

- $\left\|\partial v_\theta / \partial t\right\|$：相邻两步速度场的差分
- $\left\|\nabla_{x^b} v_\theta\right\| \cdot \left\|v_\theta\right\|$：通过 Hutchinson 随机迹估计或有限差分近似
- 或者直接利用 §4 中的代理指标进行更高效的估计

---

## §4 曲率的可观测代理：终点预测漂移

§3 定义了 token-level 局部曲率 $\kappa_i(t)$ 作为理论核心量。然而在实际推理中，精确计算 $\kappa_i(t)$ 需要额外的雅可比信息，开销较大。本节推导一个**无需额外前向传播即可观测的代理指标 (Observable Proxy)**，并证明它与 $\kappa_i(t)$ 之间存在严格的比例关系。

### §4.1 单步终点预测 (One-step Endpoint Prediction)

FM 的推理过程中，在任意时刻 $t$ 可以做一个"如果从此刻开始速度场不变，最终会到达哪里"的估计：

$$
\hat{x}_{1,i}(t) = x_i(t) + (1 - t) \cdot v_{\theta,i}(x(t), t) \tag{4.1}
$$

这就是从当前状态出发、以当前速度场做一步欧拉外推到 $t=1$ 的终点预测。当轨迹为完美直线时，$\hat{x}_{1,i}(t)$ 对所有 $t$ 恒定不变；轨迹越弯曲，不同时刻的终点预测偏差越大。

### §4.2 终点预测漂移与曲率的关系

实现中，monitor 只在 anchor step 上更新，因此我们实际比较的是相邻两次 anchor observation 之间的**终点预测漂移 (Endpoint Prediction Drift)**。记 anchor 序列为 $\{t_{a_m}\}_{m=1}^{M}$，则

$$
\hat{\delta}_{i}^{(m)} \triangleq \hat{x}_{1,i}(t_{a_m}) - \hat{x}_{1,i}(t_{a_{m-1}}) \tag{4.2}
$$

利用微分近似：

$$
\hat{\delta}_{i}^{(m)} \approx \frac{d\hat{x}_{1,i}}{dt}\bigg|_{t=t_{a_m}} \cdot \Delta t_{a_m} \tag{4.3}
$$

对式 (4.1) 求全时间导数：

$$
\frac{d\hat{x}_{1,i}}{dt} = \frac{dx_i}{dt} + (1-t) \cdot \frac{dv_{\theta,i}}{dt} - v_{\theta,i} = v_{\theta,i} + (1-t) \cdot \frac{dv_{\theta,i}}{dt} - v_{\theta,i} = (1-t) \cdot \frac{dv_{\theta,i}}{dt} \tag{4.4}
$$

因此：

$$
\hat{\delta}_{i}^{(m)} \approx (1 - t_{a_m}) \cdot \frac{dv_{\theta,i}}{dt}\bigg|_{t_{a_m}} \cdot \Delta t_{a_m} \tag{4.5}
$$

取范数：

$$
\boxed{\;\left\|\hat{\delta}_{i}^{(m)}\right\| \approx \kappa_i(t_{a_m}) \cdot (1 - t_{a_m}) \cdot \Delta t_{a_m}\;} \tag{4.6}
$$

这揭示了**终点预测漂移与局部曲率之间的严格比例关系**：

$$
\text{终点预测漂移} \;\propto\; \text{局部曲率} \;\times\; \text{剩余时间} \;\times\; \text{步长}
$$

其中 $\Delta t_{a_m} \triangleq |t_{a_m} - t_{a_{m-1}}|$。因此，$\left\|\hat{\delta}_{i}^{(m)}\right\| / \big((1 - t_{a_m})\,\Delta t_{a_m}\big)$ 可以作为 $\kappa_i(t_{a_m})$ 的无额外开销的实时估计量——它仅需两次已有的速度场评估结果。

### §4.3 Cache Granularity 设计：全局 Block Size 搜索

实现中，monitor 输出的并不是直接可调度的 block 曲率，而是 token-space 曲率图

$$
\kappa \in \mathbb{R}^{F_t \times H_t \times W_t}
$$

随后，partitioner 在候选集合

$$
\mathcal{B} = \{(b_f,b_h,b_w)\}
$$

上搜索一个全局 block size。对任意候选 $\mathbf{b} \in \mathcal{B}$，定义其诱导的均匀分区为

$$
\mathcal{P}(\mathbf{b}) = \{B_j(\mathbf{b})\}_{j=1}^{N(\mathbf{b})}
$$

其中每个 block 都是 token 网格上的轴对齐长方体，边界块允许尺寸小于 $(b_f,b_h,b_w)$。

我们的设计原则不是"让块尽可能细"，而是**让同一 block 内的曲率尽可能一致，同时保持 block 尽可能大**，以减少调度与 cache 管理开销。为此，对候选 $\mathbf{b}$ 的每个 block 定义块内离散度

$$
\mathrm{cv}_j(\mathbf{b}) \triangleq \frac{\mathrm{Std}\!\left(\{\kappa_i\}_{i \in B_j(\mathbf{b})}\right)}{\mathrm{Mean}\!\left(\{\kappa_i\}_{i \in B_j(\mathbf{b})}\right) + \varepsilon} \tag{4.7}
$$

并聚合得到

$$
\mu_{\mathrm{cv}}(\mathbf{b}) \triangleq \frac{1}{N(\mathbf{b})}\sum_{j=1}^{N(\mathbf{b})} \mathrm{cv}_j(\mathbf{b}), \qquad
q_{0.9,\mathrm{cv}}(\mathbf{b}) \triangleq \mathrm{Quantile}_{0.9}\!\left(\{\mathrm{cv}_j(\mathbf{b})\}_{j=1}^{N(\mathbf{b})}\right) \tag{4.8}
$$

实现中的选择规则可形式化为：

$$
\mathbf{b}^* =
\begin{cases}
\displaystyle \arg\max_{\mathbf{b} \in \mathcal{F}} \;\mathrm{Vol}(\mathbf{b}),
& \mathcal{F} \neq \varnothing \\[6pt]
\displaystyle \arg\min_{\mathbf{b} \in \mathcal{B}}
\Big(q_{0.9,\mathrm{cv}}(\mathbf{b}),\; \mu_{\mathrm{cv}}(\mathbf{b}),\; -\mathrm{Vol}(\mathbf{b})\Big),
& \mathcal{F} = \varnothing
\end{cases} \tag{4.9}
$$

其中可行集合

$$
\mathcal{F} \triangleq \left\{
\mathbf{b} \in \mathcal{B}\; \middle|\;
q_{0.9,\mathrm{cv}}(\mathbf{b}) \le \gamma,\;
n_{\min} \le \mathrm{Vol}(\mathbf{b}) \le n_{\max}
\right\}
$$

由块内一致性阈值 $\gamma$ 和 token 数约束共同定义。

该规则对应实现中的两个关键偏好：

- 若存在"大多数块内部已足够一致"的候选，则优先选择**最大的** block size；
- 若所有候选都不够理想，则选择 **p90-CV 最小** 的方案，并以 mean-CV 和 block 体积作 tie-break。

因此，Cubic 的 partitioner 实际优化的是**cache granularity**，而不是对单个 token 单独赋予更新周期。

### §4.4 Block Representative Curvature

在确定分区 $\mathcal{P}(\mathbf{b}^*)$ 后，optimizer 不直接使用块内均值，而是为每个 block 定义一个偏保守的代表曲率：

$$
\kappa_b \triangleq \mathrm{Quantile}_{q}\!\left(\{\kappa_i\}_{i \in B_b}\right), \qquad q \approx 0.9 \tag{4.10}
$$

相比均值聚合，分位数聚合更能抑制"大块中少量高曲率 token 被平均掉"的问题，从而减少过度跳步的风险。直观上，$\kappa_b$ 更接近该 block 的"risk-aware curvature"。

---

## §5 误差建模：零阶保持器 (ZOH) 截断误差

有了 $\kappa_b$ 作为局部曲率的量化，我们现在严谨地建模"跳步"（复用缓存的速度场）所引入的误差。

### §5.1 ZOH 策略的形式化

假设在时刻 $t_0$ 对块 $b$ 进行了一次精确的速度场计算：

$$
v_0 = v_\theta(x(t_0),\, t_0)
$$

在接下来连续 $\tau$ 个离散步长（时间区间 $[t_0,\, t_0 + \tau\Delta t]$）内，我们不再重新计算块 $b$ 的速度场，而是复用 $v_0$：

$$
\text{近似 (ZOH) 轨道:} \quad \frac{d\tilde{x}}{dt} = v_0, \quad \tilde{x}(t_0) = x(t_0)
$$

$$
\text{真实 (Exact) 轨道:} \quad \frac{dx}{dt} = v_\theta(x(t),\, t), \quad x(t_0) = x(t_0)
$$

这等价于对 ODE 施加零阶保持器 (Zero-Order Hold)，也等价于用单步前向欧拉 (Forward Euler) 跨越 $\tau$ 个子步。当轨迹为直线（$\kappa = 0$）时，ZOH 无误差；当轨迹弯曲（$\kappa > 0$）时，ZOH 将偏离真实轨迹。

### §5.2 局部截断误差推导

为避免与前文的终点预测漂移 $\hat{\delta}_{i}^{(m)}$ 混淆，我们将 ZOH 引入的状态误差记为 $e(t)$。定义

$$
e(t) \triangleq \tilde{x}(t) - x(t), \qquad e(t_0)=0
$$

则误差演化方程为：

$$
\frac{de}{dt} = v_0 - v_\theta(x(t),\, t) \tag{5.1}
$$

对右侧，利用真实轨道上速度场全导数的积分形式：

$$
v_0 - v_\theta(x(t),\, t) = -\int_{t_0}^{t} \frac{dv_\theta}{ds}(x(s),\, s)\, ds \tag{5.2}
$$

取范数，利用 $\kappa_b$ 的定义作为全导数范数的上界：

$$
\left\|v_0 - v_\theta(x(t),\, t)\right\| \le \int_{t_0}^{t} \kappa_b(s)\, ds \tag{5.3}
$$

在 $\tau\Delta t$ 窗口足够短的条件下，可将 $\kappa_b(s)$ 近似为区间内的常数 $\kappa_b$：

$$
\left\|v_0 - v_\theta(x(t),\, t)\right\| \le \kappa_b \cdot (t - t_0) \tag{5.4}
$$

对误差演化方程 (5.1) 积分：

$$
\|e(t)\| \le \int_{t_0}^{t} \kappa_b \cdot (s - t_0)\, ds = \frac{1}{2}\,\kappa_b\,(t - t_0)^2 \tag{5.5}
$$

令 $t = t_0 + \tau\Delta t$，得到跳过 $\tau$ 步后的**局部截断误差上界**：

$$
\boxed{\;\|e_\tau\| \le \frac{1}{2}\,\kappa_b \cdot \tau^2 \Delta t^2\;} \tag{5.6}
$$

**物理直觉：**

- 误差与曲率 $\kappa_b$ **线性**相关——轨迹越弯，误差越大
- 误差与跳步数 $\tau$ 的**平方**相关——多跳一步的代价递增
- 误差与步长 $\Delta t$ 的**平方**相关——与欧拉法的一阶截断一致

### §5.3 关于 Lipschitz 修补与 Grönwall 累积

上述推导 (5.4) 中直接使用了真实轨道上的速度场偏离来约束误差增长。严格而言，ZOH 系统实际运行在近似轨道 $\tilde{x}(t)$ 上，与真实轨道 $x(t)$ 存在偏离，这会引入二阶修正项。

若速度场满足空间 Lipschitz 条件（常数 $L$）：

$$
\|v_\theta(\tilde{x},\, t) - v_\theta(x,\, t)\| \le L\,\|e(t)\|
$$

则更精确的误差演化为：

$$
\frac{d\|e\|}{dt} \le \kappa_b\,(t - t_0) + L\,\|e(t)\|
$$

由 Grönwall 不等式：

$$
\|e(t)\| \le \frac{\kappa_b}{L}\left[e^{L(t - t_0)} - 1 - L(t - t_0)\right]
$$

当 $L \cdot \tau\Delta t \ll 1$ 时（短窗口），上式退化为 $\frac{1}{2}\kappa_b(\tau\Delta t)^2$，与 (5.6) 一致。在我们关注的加速比范围内（$\tau$ 通常为 2–8），该条件成立，因此 (5.6) 作为工作误差界已足够精确。

### §5.4 曲率稳定性假设 (Curvature Stationarity)

在生成过程的不同阶段，$\kappa_b$ 的行为有所不同：

- **初期** ($t \approx 0$)：从纯噪声出发，全局结构尚未建立，$\kappa_b$ 波动较大
- **中期** ($t \in [0.2, 0.8]$)：整体轮廓已确定，细节逐步填充，$\kappa_b$ 趋于稳定——这是加速的主要作用区间
- **末期** ($t \approx 1$)：最终细节修饰，$\kappa_b$ 可能再次升高

> **假设 2 (Curvature Stationarity)：** 在中期区间 $[t_a, t_b]$ 内，各块的曲率 $\kappa_b$ 可近似视为与时间无关的常数。该常数可由前期的 anchor 观测历史估计得到：
>
> $$\kappa_b \approx \mathrm{median}_m \left\{ \mathrm{Agg}_b\!\left(\frac{\|\hat{\delta}_{i}^{(m)}\|}{(1 - t_{a_m})\,\Delta t_{a_m}}\right) \right\}, \quad t_{a_m} < t_a$$

初期和末期保持完整计算（所有块均更新），仅在中期应用自适应调度。

---

## §6 代理损失与优化目标

### §6.1 代理损失定义

FM 的训练目标为 MSE 损失，因此我们以状态误差的平方作为代理损失。对块 $b$ 分配跳步间隔 $\tau_b$ 所产生的损失为：

$$
\ell_b(\tau_b) \triangleq \|e_{\tau_b}\|^2 \le \left(\frac{1}{2}\,\kappa_b\,\tau_b^2\,\Delta t^2\right)^2 = \frac{1}{4}\,\kappa_b^2\,\Delta t^4\,\tau_b^4 \tag{6.1}
$$

即：

$$
\ell_b(\tau_b) \;\propto\; \kappa_b^2 \cdot \tau_b^4 \tag{6.2}
$$

### §6.2 关于幂次指数的讨论

理论推导给出 $\ell_b \propto \tau_b^4$（误差 $\propto \tau^2$，代理损失 = 误差²）。然而：

- **(a)** 若采用 $L_1$ 损失度量（误差本身而非平方），则 $\ell_b \propto \tau_b^2$
- **(b)** 若考虑感知质量（如 FID, LPIPS），误差与感知质量的映射可能是非线性的
- **(c)** Grönwall 累积效应在多次跳步时可能修改有效指数

因此我们将代理损失参数化为：

$$
\ell_b(\tau_b) \;\propto\; \kappa_b^\alpha \cdot \tau_b^\beta \tag{6.3}
$$

理论预测 $\alpha = 2,\; \beta = 4$。实际实现中可通过小规模实验校准 $\beta$ 的取值。后续推导以一般 $(\alpha, \beta)$ 进行，最终代入理论值。

---

## §7 Block-wise 自适应更新调度

### §7.1 优化问题

给定 $B$ 个块，每块具有曲率 $\kappa_b$，需要为其分配更新间隔 $\tau_b$。目标：在给定全局计算预算下，最小化所有块的总代理损失。

$$
\min_{\tau_b \ge 1} \;\sum_{b=1}^{B} \kappa_b^\alpha \cdot \tau_b^\beta \qquad \text{s.t.} \quad \frac{1}{B}\sum_{b=1}^{B} \frac{1}{\tau_b} = \frac{1}{\rho} \tag{7.1}
$$

其中 $\rho$ 为目标加速比。约束条件的含义：$1/\tau_b$ 是块 $b$ 在每步中的"计算概率"（每 $\tau_b$ 步计算一次），所有块的平均计算概率 $= 1/\rho$，即平均每步仅更新 $B/\rho$ 个块。

### §7.2 KKT 求解

构造 Lagrangian：

$$
\mathcal{L} = \sum_{b=1}^{B} \kappa_b^\alpha\,\tau_b^\beta + \lambda\left(\frac{1}{B}\sum_{b=1}^{B} \frac{1}{\tau_b} - \frac{1}{\rho}\right)
$$

对 $\tau_b$ 求偏导令其为零：

$$
\frac{\partial \mathcal{L}}{\partial \tau_b} = \beta\,\kappa_b^\alpha\,\tau_b^{\beta - 1} - \frac{\lambda}{B\,\tau_b^2} = 0
$$

整理：

$$
\tau_b^{\beta + 1} = \frac{\lambda}{\beta\,B\,\kappa_b^\alpha}
$$

$$
\Longrightarrow \quad \tau_b^* \;\propto\; \kappa_b^{-\alpha/(\beta+1)} \tag{7.2}
$$

代入理论值 $\alpha = 2,\; \beta = 4$：

$$
\boxed{\;\tau_b^* \;\propto\; \kappa_b^{-2/5}\;} \tag{7.3}
$$

**直觉：** 曲率高（直线度低）的块获得更小的 $\tau$（更频繁更新），曲率低（直线度高）的块获得更大的 $\tau$（更少更新，更多跳步）。

### §7.3 闭式调度公式

将比例关系 (7.2) 代入约束条件，解出绝对值：

$$
\tilde{\tau}_b = \rho \cdot \frac{\overline{\kappa^{\alpha/(\beta+1)}}}{\kappa_b^{\alpha/(\beta+1)}} \tag{7.4}
$$

其中

$$
\overline{\kappa^{\alpha/(\beta+1)}} \triangleq \frac{1}{B}\sum_{b'=1}^{B} \kappa_{b'}^{\alpha/(\beta+1)}
$$

代入 $\alpha = 2,\; \beta = 4$：

$$
\tilde{\tau}_b = \rho \cdot \frac{\overline{\kappa^{2/5}}}{\kappa_b^{2/5}}, \qquad \overline{\kappa^{2/5}} = \frac{1}{B}\sum_{b'=1}^{B} \kappa_{b'}^{2/5} \tag{7.5}
$$

### §7.4 离散化与约束修正

实际中 $\tau_b$ 必须为正整数，且不得小于 1。处理策略：

$$
\tau_b^* = \text{Round}\!\left(\max\!\left(1,\;\; \rho' \cdot \frac{\overline{\kappa^{2/5}}}{\kappa_b^{2/5}}\right)\right) \tag{7.6}
$$

其中 $\rho' \approx \rho$ 是经截断补偿后的有效加速比。具体而言：

1. 先计算连续解 $\tilde{\tau}_b$
2. 对 $\tilde{\tau}_b < 1$ 的块截断为 1（高曲率块至少每步更新一次）
3. 将截断后剩余的预算重新分配给其他块
4. 四舍五入为整数
5. 微调以精确满足预算约束

实际并不要求精确达到目标加速比 $\rho$，允许在 $[\rho - \varepsilon,\; \rho + \varepsilon]$ 范围内浮动。

---

### §7.5 Selective Token-wise DiT Forward

当分区 $\mathcal{P}$ 与 block interval $\{\tau_b\}$ 确定后，第 $k$ 个 selective step 的 active block 集合为

$$
\mathcal{A}_k \triangleq \{b \in \mathcal{P} \mid s_k \bmod \tau_b = 0\} \tag{7.7}
$$

其中 $s_k$ 是 selective phase 内的局部步编号。对应的 active / frozen token 集合为

$$
\Omega_k^{\mathrm{act}} \triangleq \bigcup_{b \in \mathcal{A}_k} B_b, \qquad
\Omega_k^{\mathrm{frz}} \triangleq \Omega \setminus \Omega_k^{\mathrm{act}} \tag{7.8}
$$

在 transformer 第 $\ell$ 层，Cubic 只对 active token 重新计算 query，并用 active token 的新 K/V 更新 per-layer cache；frozen token 则直接复用历史 cache：

$$
Q_\ell^{\mathrm{act}} = W_\ell^Q h_\ell(\Omega_k^{\mathrm{act}}), \qquad
K_\ell^{\mathrm{act}} = W_\ell^K h_\ell(\Omega_k^{\mathrm{act}}), \qquad
V_\ell^{\mathrm{act}} = W_\ell^V h_\ell(\Omega_k^{\mathrm{act}}) \tag{7.9}
$$

$$
K_\ell^{\mathrm{full}} = \mathrm{Scatter}\!\left(K_\ell^{\mathrm{act}}, K_{\ell,\mathrm{cache}}^{\mathrm{frz}}\right), \qquad
V_\ell^{\mathrm{full}} = \mathrm{Scatter}\!\left(V_\ell^{\mathrm{act}}, V_{\ell,\mathrm{cache}}^{\mathrm{frz}}\right) \tag{7.10}
$$

随后 attention 仅为 active token 产出输出：

$$
Y_\ell^{\mathrm{act}} = \mathrm{Attn}\!\left(Q_\ell^{\mathrm{act}}, K_\ell^{\mathrm{full}}, V_\ell^{\mathrm{full}}\right) \tag{7.11}
$$

并且 FFN / head 也仅在 active token 上执行；冻结 token 的层输出或最终速度预测从 output cache 中直接取回。于是一步 selective forward 的输出可写成

$$
v_k(\Omega) =
\mathrm{Merge}\!\left(
v_k^{\mathrm{act}}(\Omega_k^{\mathrm{act}}),
v_{k-1,\mathrm{cache}}(\Omega_k^{\mathrm{frz}})
\right) \tag{7.12}
$$

实际工程中，当 KV cache 或 output cache 不可用时，系统回退到 full forward 以刷新缓存。这一点保证了 selective path 的数值稳定性，也使理论中的 ZOH 近似在实现上具备明确的 bootstrap / refresh 机制。

## §8 总结：理论框架的逻辑链条

```
1. Rectified Flow 旨在拉直生成轨迹，使 dv/dt → 0
   但实际模型的轨迹并非完美直线，残余曲率空间分布不均
                              ↓
2. 定义 Block-wise 局部曲率 κ_b(t) = ‖dv_θ^{(b)}/dt‖
   并证明可通过终点预测漂移 ‖δ̂^{(b,k)}‖ 无额外开销地估计
                              ↓
3. ZOH（跳步/缓存复用）的局部截断误差 ∝ κ_b · τ²Δt²
   曲率直接决定了步长容忍度
                              ↓
4. Anchor step 上先搜索全局 block size，使块内曲率尽量一致且块尽量大
   再以 block 分位数曲率作为风险感知代表量
                              ↓
5. 给定总计算预算（加速比 ρ），最优调度为
   τ_b* ∝ κ_b^{-2/5}
   曲率低（直）的块少算，曲率高（弯）的块多算
                              ↓
6. 实现为在线自适应算法：
   前期全量计算并估计各块 κ_b
   anchor 时重估粒度与 τ_b，中期间 selective token forward
   末期恢复全量计算保证最终质量
```

---

## 附录 A：关键假设汇总

**假设 1 (Spatial Locality)：** 短时间步内，跨块雅可比项可忽略，$\kappa_b$ 仅由本块状态决定。成立条件：DiT attention 的空间稀疏性 + $\Delta t$ 足够小。

**假设 2 (Curvature Stationarity)：** 生成中期，各块的 $\kappa_b$ 近似为时间常数。成立条件：轮廓已建立、细节尚未进入最终修饰阶段。

**假设 3 (Error Independence)：** 各块的局部截断误差独立累积，忽略跨块注意力引起的误差传播。成立条件：与假设 1 同源，在短窗口内由 Lipschitz 条件保证。

---

## 附录 B：与现有工作的理论关系

### B.1 与 Rectified Flow 的 Straightness 指标

Rectified Flow (Liu et al., 2022) 定义全局 straightness 为 $S = \mathbb{E}_t\!\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$。我们的 $\kappa_b(t)$ 是该概念在空间和时间上的局部化版本。当 $\kappa_b = 0$ 对所有 $b$ 成立时，$S = 0$（完美直线）。Reflow 操作全局降低 $S$，而 Cubic 利用 $S$ 的空间异质性进行局部调度。

### B.2 与自适应 ODE 求解器的关系

传统自适应 ODE 求解器（如 Dormand-Prince / RK45）根据全局截断误差估计来调整时间步长。Cubic 可以被理解为一种 **spatially-adaptive ODE solver**：在同一时间步内，对不同空间区域使用不同的有效步长。

### B.3 与 DeepCache 的关系

DeepCache (Ma et al., 2023) 利用相邻时间步的特征相似性进行缓存，但其缓存策略是全局统一的（所有空间区域相同的跳步间隔）。Cubic 可以视为 DeepCache 的理论驱动、曲率自适应、空间细粒度升级版。

---

## 附录 C：实验校准指南

理论预测的关键可校准参数：

- **$\beta$ 的取值：** 理论值 4，建议在 $\{2, 3, 4, 5\}$ 中搜索，以 FID/LPIPS vs. 加速比的帕累托前沿为判据。
- **候选 block size 集合 $\mathcal{B}$：** 影响 partitioner 的搜索空间。实现中更合理的调参方式不是手工固定某个 $(f,h,w)$，而是调节 `min_block_size / max_block_size / cv_threshold / token-count` 约束，让系统在候选中自动选择全局 block size。
- **块代表曲率的聚合方式：** 当前实现采用 block 内高分位数（默认 q=0.9），比均值更保守。若希望更激进的加速，可尝试降低 q；若希望降低局部 artifact 风险，可提高 q。
- **加速区间 $[t_a, t_b]$：** 中期稳定区间的边界。建议通过观察 $\kappa_b(t)$ 的时间曲线确定，典型值 $t_a \approx 0.15$，$t_b \approx 0.85$。
- **$\kappa_b$ 的估计窗口：** 用多少步的历史来估计中期 $\kappa_b$。建议取 $t < t_a$ 阶段的 median 或 exponential moving average。
