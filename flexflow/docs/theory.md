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

（对于图像 $F=1$）。我们对时空维度 $(f, h, w)$ 进行均匀分块，将张量划分为 $B$ 个不重叠的块 $\{b\}_{1}^{B}$。

**经验观察 (Empirical Observation)：** 在固定总步数 $K$ 的采样过程中，不同块的生成轨迹表现出显著的**异质性 (Spatial Heterogeneity)**——

- 纹理平滑、变化缓慢的区域（如背景、大面积色块），其速度场沿时间方向几乎恒定，即轨迹近似直线，仅需很少的步数即可达到视觉收敛；
- 纹理复杂、高频细节密集的区域（如人脸、文字、主体边缘），其速度场持续调整方向，即轨迹弯曲程度大，需要更多步数才能忠实还原。

这一现象的本质并非"某些区域提前收敛"，而是**不同空间区域的生成轨迹具有不同的直线度 (Straightness)**。直线度高的区域对大步长（粗粒度离散化）的容忍度高；直线度低的区域则要求小步长以跟踪弯曲的轨迹。

**核心思路：**

> **观察** → 内容复杂度与收敛速度强相关
> **分析** → 底层原因是局部流直线度 (Local Flow Straightness) 决定了步长容忍度 (Step-size Tolerance)
> **设计** → 基于 Block-wise Flow Straightness 的自适应调度，对不同块分配不同的更新频率（ZOH 缓存策略）

---

## §3 局部轨迹曲率：定义与分解

### §3.1 轨迹曲率的定义

对于块 $b$，其在真实 ODE 轨迹上的速度场随时间的**全导数**为：

$$
\frac{d v_\theta^{(b)}}{dt}(t) = \frac{\partial v_\theta^{(b)}}{\partial t} + \left(\nabla_{x^b} v_\theta^{(b)}\right) \cdot v_\theta^{(b)} \tag{3.1}
$$

其中右侧第一项是速度场的**显式时间变化率 (Explicit temporal rate)**，第二项是**对流传输项 (Advective transport)**，度量了因状态 $x^b$ 沿轨迹移动而引起的速度场隐式变化。

我们将块 $b$ 在时刻 $t$ 的**局部轨迹曲率 (Local Trajectory Curvature)** 定义为该全导数的范数：

$$
\kappa_b(t) \triangleq \left\| \frac{d v_\theta^{(b)}}{dt}(t) \right\| \tag{3.2}
$$

**直观含义：**

- $\kappa_b(t) = 0$ ⟹ 速度场在块 $b$ 上恒定，轨迹为直线，**完美直线度**
- $\kappa_b(t)$ 大 ⟹ 速度场在块 $b$ 上剧烈变化，轨迹弯曲，**直线度低**

这与 Rectified Flow 的 "straightness" 概念完全一致：Rectified Flow 的 reflow 操作旨在全局降低 $\kappa$，而我们的方法则利用 $\kappa$ 的**空间异质性**进行局部自适应调度。

### §3.2 空间局部性假设 (Spatial Locality Assumption)

在 DiT 等具有全局自注意力的架构中，块 $b$ 的速度场严格来说依赖于所有块的状态。因此全导数应展开为：

$$
\frac{d v_\theta^{(b)}}{dt} = \frac{\partial v_\theta^{(b)}}{\partial t} + \sum_{b'} \frac{\partial v_\theta^{(b)}}{\partial x^{(b')}} \cdot \frac{d x^{(b')}}{dt} \tag{3.3}
$$

为使理论在块级别闭环，我们引入空间局部性假设：

> **假设 1 (Spatial Locality)：** 在时间步 $\Delta t$ 内，跨块雅可比项 $\partial v_\theta^{(b)} / \partial x^{(b')}$（$b' \neq b$）相对于同块自雅可比项可忽略，从而
>
> $$\frac{d v_\theta^{(b)}}{dt} \approx \frac{\partial v_\theta^{(b)}}{\partial t} + \left(\nabla_{x^b} v_\theta^{(b)}\right) \cdot v_\theta^{(b)}$$

该假设的合理性依据：

- **(a)** 多项研究表明 DiT 的 self-attention score 表现出较强的空间局部性和稀疏性，远距离 token 间的注意力权重衰减迅速。
- **(b)** 在短时间窗口 $\Delta t$ 内，跨块状态变化为 $O(\Delta t)$ 量级，其对本块速度场的扰动为二阶小量 $O(\Delta t^2)$，在一阶分析中自然消去。
- **(c)** 实验验证：将跨块雅可比冻结与否对比，误差差异在统计意义上不显著。

> **注：** 严格而言，式 (3.3) 中的完整全导数可以通过向量-雅可比积 (VJP) 直接计算，不存在计算上的障碍。空间局部性假设的目的是简化理论分析，使 $\kappa_b(t)$ 成为仅依赖块 $b$ 本身状态的局部量，从而支撑后续的独立块调度。

### §3.3 利用三角不等式建立可计算上界

对式 (3.2) 应用三角不等式：

$$
\kappa_b(t) \le \left\|\frac{\partial v_\theta}{\partial t}\right\| + \left\|\nabla_{x^b} v_\theta\right\| \cdot \left\|v_\theta\right\| \tag{3.4}
$$

右侧的三个量均可在推理过程中估计：

- $\left\|\partial v_\theta / \partial t\right\|$：相邻两步速度场的差分
- $\left\|\nabla_{x^b} v_\theta\right\| \cdot \left\|v_\theta\right\|$：通过 Hutchinson 随机迹估计或有限差分近似
- 或者直接利用 §4 中的代理指标进行更高效的估计

---

## §4 曲率的可观测代理：终点预测漂移

§3 定义了局部曲率 $\kappa_b(t)$ 作为理论核心量。然而在实际推理中，精确计算 $\kappa_b(t)$ 需要额外的雅可比信息，开销较大。本节推导一个**无需额外前向传播即可观测的代理指标 (Observable Proxy)**，并证明它与 $\kappa_b(t)$ 之间存在严格的比例关系。

### §4.1 单步终点预测 (One-step Endpoint Prediction)

FM 的推理过程中，在任意时刻 $t$ 可以做一个"如果从此刻开始速度场不变，最终会到达哪里"的估计：

$$
\hat{x}_1^{(b)}(t) = x^b(t) + (1 - t) \cdot v_\theta(x^b(t), t) \tag{4.1}
$$

这就是从当前状态出发、以当前速度场做一步欧拉外推到 $t=1$ 的终点预测。当轨迹为完美直线时，$\hat{x}_1^{(b)}(t)$ 对所有 $t$ 恒定不变；轨迹越弯曲，不同时刻的终点预测偏差越大。

### §4.2 终点预测漂移与曲率的关系

定义相邻步之间的**终点预测漂移 (Endpoint Prediction Drift)**：

$$
\hat{\delta}^{(b,k)} \triangleq \hat{x}_1^{(b)}(t_k) - \hat{x}_1^{(b)}(t_{k-1}) \tag{4.2}
$$

利用微分近似：

$$
\hat{\delta}^{(b,k)} \approx \frac{d\hat{x}_1^{(b)}}{dt}\bigg|_{t=t_k} \cdot \Delta t \tag{4.3}
$$

对式 (4.1) 求全时间导数：

$$
\frac{d\hat{x}_1^{(b)}}{dt} = \frac{dx^b}{dt} + (1-t) \cdot \frac{dv_\theta}{dt} - v_\theta = v_\theta + (1-t) \cdot \frac{dv_\theta}{dt} - v_\theta = (1-t) \cdot \frac{dv_\theta}{dt} \tag{4.4}
$$

因此：

$$
\hat{\delta}^{(b,k)} \approx (1 - t_k) \cdot \frac{dv_\theta^{(b)}}{dt}\bigg|_{t_k} \cdot \Delta t \tag{4.5}
$$

取范数：

$$
\boxed{\;\left\|\hat{\delta}^{(b,k)}\right\| \approx \kappa_b(t_k) \cdot (1 - t_k) \cdot \Delta t\;} \tag{4.6}
$$

这揭示了**终点预测漂移与局部曲率之间的严格比例关系**：

$$
\text{终点预测漂移} \;\propto\; \text{局部曲率} \;\times\; \text{剩余时间} \;\times\; \text{步长}
$$

因此，$\left\|\hat{\delta}^{(b,k)}\right\| / \big((1 - t_k)\,\Delta t\big)$ 可以作为 $\kappa_b(t_k)$ 的无额外开销的实时估计量——它仅需两次已有的速度场评估结果。

### §4.3 Blockwise Straightness Score

为了提供一个归一化的、跨时刻可比较的指标，我们定义块 $b$ 在时刻 $t_k$ 的**直线度分数 (Straightness Score)**：

$$
S_b(t_k) \triangleq 1 - \frac{\left\|\hat{\delta}^{(b,k)}\right\|}{(1 - t_k) \cdot \left\|v_\theta\right\| \cdot \Delta t} \tag{4.7}
$$

其中分母中的 $\left\|v_\theta\right\|$ 提供尺度归一化。

- $S_b \approx 1$ ⟹ 轨迹近似直线，曲率低，可以安全跳步
- $S_b \approx 0$ ⟹ 轨迹弯曲剧烈，曲率高，需要密集更新

> **注：** 在实际实现中，直接使用 $\kappa_b$ 的估计值或 $\|\hat{\delta}\|$ 的原始值已经足够，$S_b$ 仅在可视化和跨实验比较时提供便利。后续理论推导统一使用 $\kappa_b$。

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

定义状态误差 $\delta(t) \triangleq \tilde{x}(t) - x(t)$，初始条件 $\delta(t_0) = 0$。误差演化方程：

$$
\frac{d\delta}{dt} = v_0 - v_\theta(x(t),\, t) \tag{5.1}
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
\|\delta(t)\| \le \int_{t_0}^{t} \kappa_b \cdot (s - t_0)\, ds = \frac{1}{2}\,\kappa_b\,(t - t_0)^2 \tag{5.5}
$$

令 $t = t_0 + \tau\Delta t$，得到跳过 $\tau$ 步后的**局部截断误差上界**：

$$
\boxed{\;\|\delta_\tau\| \le \frac{1}{2}\,\kappa_b \cdot \tau^2 \Delta t^2\;} \tag{5.6}
$$

**物理直觉：**

- 误差与曲率 $\kappa_b$ **线性**相关——轨迹越弯，误差越大
- 误差与跳步数 $\tau$ 的**平方**相关——多跳一步的代价递增
- 误差与步长 $\Delta t$ 的**平方**相关——与欧拉法的一阶截断一致

### §5.3 关于 Lipschitz 修补与 Grönwall 累积

上述推导 (5.4) 中直接使用了真实轨道上的速度场偏离来约束误差增长。严格而言，ZOH 系统实际运行在近似轨道 $\tilde{x}(t)$ 上，与真实轨道 $x(t)$ 存在偏离，这会引入二阶修正项。

若速度场满足空间 Lipschitz 条件（常数 $L$）：

$$
\|v_\theta(\tilde{x},\, t) - v_\theta(x,\, t)\| \le L\,\|\delta(t)\|
$$

则更精确的误差演化为：

$$
\frac{d\|\delta\|}{dt} \le \kappa_b\,(t - t_0) + L\,\|\delta(t)\|
$$

由 Grönwall 不等式：

$$
\|\delta(t)\| \le \frac{\kappa_b}{L}\left[e^{L(t - t_0)} - 1 - L(t - t_0)\right]
$$

当 $L \cdot \tau\Delta t \ll 1$ 时（短窗口），上式退化为 $\frac{1}{2}\kappa_b(\tau\Delta t)^2$，与 (5.6) 一致。在我们关注的加速比范围内（$\tau$ 通常为 2–8），该条件成立，因此 (5.6) 作为工作误差界已足够精确。

### §5.4 曲率稳定性假设 (Curvature Stationarity)

在生成过程的不同阶段，$\kappa_b$ 的行为有所不同：

- **初期** ($t \approx 0$)：从纯噪声出发，全局结构尚未建立，$\kappa_b$ 波动较大
- **中期** ($t \in [0.2, 0.8]$)：整体轮廓已确定，细节逐步填充，$\kappa_b$ 趋于稳定——这是加速的主要作用区间
- **末期** ($t \approx 1$)：最终细节修饰，$\kappa_b$ 可能再次升高

> **假设 2 (Curvature Stationarity)：** 在中期区间 $[t_a, t_b]$ 内，各块的曲率 $\kappa_b$ 可近似视为与时间无关的常数。该常数可由前期（如 $t < t_a$）的速度场评估历史估计得到：
>
> $$\kappa_b \approx \text{median}_k \left\{ \frac{\|\hat{\delta}^{(b,k)}\|}{(1 - t_k)\,\Delta t} \right\}, \quad t_k < t_a$$

初期和末期保持完整计算（所有块均更新），仅在中期应用自适应调度。

---

## §6 代理损失与优化目标

### §6.1 代理损失定义

FM 的训练目标为 MSE 损失，因此我们以状态误差的平方作为代理损失。对块 $b$ 分配跳步间隔 $\tau_b$ 所产生的损失为：

$$
\ell_b(\tau_b) \triangleq \|\delta_{\tau_b}\|^2 \le \left(\frac{1}{2}\,\kappa_b\,\tau_b^2\,\Delta t^2\right)^2 = \frac{1}{4}\,\kappa_b^2\,\Delta t^4\,\tau_b^4 \tag{6.1}
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
4. 给定总计算预算（加速比 ρ），最优调度为
   τ_b* ∝ κ_b^{-2/5}
   曲率低（直）的块少算，曲率高（弯）的块多算
                              ↓
5. 实现为在线自适应算法：
   前期全量计算并估计各块 κ_b
   中期按 κ_b 分配更新频率
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
- **分块粒度 $(f, h, w)$：** 影响空间局部性假设的精度。块太大 → 块内曲率异质性高，调度粗糙；块太小 → 假设更好，但调度开销增加。建议初始值：$h = w = H/4$ 或 $H/8$。
- **加速区间 $[t_a, t_b]$：** 中期稳定区间的边界。建议通过观察 $\kappa_b(t)$ 的时间曲线确定，典型值 $t_a \approx 0.15$，$t_b \approx 0.85$。
- **$\kappa_b$ 的估计窗口：** 用多少步的历史来估计中期 $\kappa_b$。建议取 $t < t_a$ 阶段的 median 或 exponential moving average。