\section{Methodology}
\label{sec:method}

We present \cubic, a curvature-driven dynamic caching framework for Flow Matching samplers. The core idea is to connect local trajectory geometry with runtime caching decisions: regions whose trajectories are locally straighter can tolerate more aggressive cache reuse, whereas regions with higher curvature should be refreshed more frequently. To make this idea practical, \cubic decomposes inference-time control into four stages: a lightweight monitor estimates a token-wise curvature proxy online; a partitioner determines the cache granularity in token space; an optimizer maps block curvature to block-specific update intervals under a target speedup budget; and a selective forward engine executes the resulting schedule with sparse token computation.

\subsection{Overview}
\label{sec:method_overview}

We consider the Euler discretization of a Flow Matching sampler,
\begin{equation}
    x_{k+1} = x_k + \Delta t_k \, v_\theta(x_k, t_k),
\end{equation}
where $v_\theta$ denotes the learned velocity field. If the local velocity remains nearly constant along the trajectory, then extrapolation and feature reuse are both reliable; if the velocity changes rapidly, stale cached features incur larger errors. \cubic exploits this spatial heterogeneity by alternating between two types of inference steps during sampling:
\begin{itemize}
    \item \textbf{Anchor steps}: full forward passes that refresh the exact velocity prediction for all tokens, update the curvature proxy, and re-evaluate the caching policy.
    \item \textbf{Selective steps}: sparse forward passes that recompute only a subset of tokens (the active tokens), while the remaining tokens (the frozen tokens) reuse cached intermediate states and outputs.
\end{itemize}
To preserve generation quality, we divide the sampling trajectory into three phases: warmup, cruise, and cooldown. Warmup and cooldown use full computation, while the cruise phase alternates between anchor and selective steps.

\subsection{Curvature Proxy via Endpoint Prediction Drift}
\label{sec:monitor}

Directly computing local trajectory curvature is impractical at inference time because it requires Jacobian information. We therefore seek an online proxy that introduces no additional model forward passes. Our key observation is that if a token trajectory is locally straight, then the endpoint predicted by the current velocity field should remain nearly unchanged across successive observations. In contrast, trajectory curvature causes this endpoint prediction to drift.

For a token at position $i$, we define the endpoint prediction at time $t$ as
\begin{equation}
    \hat{x}_{1,i}(t) = x_i(t) + (1 - t)\, v_{\theta,i}(x(t), t).
\end{equation}
Importantly, the monitor is updated only at anchor steps rather than at every sampler step. Let $\{t_{a_m}\}_{m=1}^{M}$ denote the anchor-step sequence. We define the endpoint prediction drift between two consecutive anchor observations as
\begin{equation}
    \hat{\delta}_{i}^{(m)} = \hat{x}_{1,i}(t_{a_m}) - \hat{x}_{1,i}(t_{a_{m-1}}).
\end{equation}
As shown in the Appendix, this drift approximates the local token-wise trajectory curvature:
\begin{equation}
    \left\|\hat{\delta}_{i}^{(m)}\right\|
    \approx
    \kappa_i(t_{a_m})\,(1-t_{a_m})\,\Delta t_{a_m},
\end{equation}
where $\Delta t_{a_m} = |t_{a_m} - t_{a_{m-1}}|$ is the interval between two anchor observations. Hence, the normalized quantity $\|\hat{\delta}_{i}^{(m)}\| / \big((1-t_{a_m})\Delta t_{a_m}\big)$ serves as a zero-overhead proxy for token-wise curvature.

In practice, we compute this proxy in latent space and aggregate it to the token grid using the same patch granularity as the underlying DiT tokenizer, producing a token-space curvature map
\begin{equation}
    \kappa \in \mathbb{R}^{F_t \times H_t \times W_t}.
\end{equation}
This curvature map is the sole signal consumed by the subsequent partitioner and optimizer; the monitor itself does not directly decide which tokens should be skipped.

\subsection{Cache Granularity Design}
\label{sec:partitioner}

Given the token-space curvature map, the next question is not to assign an independent update period to every token. Instead, \cubic first determines an appropriate caching granularity, because selective execution is applied to groups of tokens that share the same caching policy.

Specifically, \cubic defines block partitions directly in token space. At each anchor step, it selects a global block size $\mathbf{b}=(b_f,b_h,b_w)$ from a candidate set and tiles the entire token grid uniformly with that size. Thus, the partition is adaptive in \emph{granularity} but uniform in \emph{tiling}.

For any candidate block size $\mathbf{b}$ that induces a partition $\mathcal{P}(\mathbf{b})=\{B_j(\mathbf{b})\}$, we measure intra-block consistency by the coefficient of variation (CV):
\begin{equation}
    \mathrm{cv}_j(\mathbf{b}) =
    \frac{\mathrm{Std}(\{\kappa_i\}_{i \in B_j(\mathbf{b})})}
    {\mathrm{Mean}(\{\kappa_i\}_{i \in B_j(\mathbf{b})}) + \varepsilon}.
\end{equation}
The selection principle is simple: choose the coarsest block size whose induced blocks remain internally curvature-consistent. In practice, this criterion is approximated using block-wise CV statistics such as mean-CV and p90-CV. If a candidate already satisfies the consistency threshold, \cubic prefers the largest block size; otherwise, it falls back to the candidate with the best consistency statistics.

Once the partition is fixed, we aggregate token-wise curvature into a representative block curvature. Rather than using the block mean, we adopt a conservative high quantile:
\begin{equation}
    \kappa_b = \mathrm{Quantile}_{q}\big(\{\kappa_i\}_{i \in B_b}\big), \qquad q \approx 0.9.
\end{equation}
This design prevents a few high-curvature tokens from being drowned out by the block average, thereby reducing the risk of over-aggressive cache reuse.

\subsection{Budget-Constrained Interval Optimization}
\label{sec:optimizer}

After obtaining a representative curvature for each block, \cubic determines how long each block may safely reuse cached outputs. The key insight is that reusing a cached block output for $\tau_b$ steps is equivalent to applying a zero-order hold to the local velocity field over a short time window. To avoid overloading the symbol used for endpoint-prediction drift, we denote the resulting ZOH truncation error by $e_b$. Our theoretical analysis in the Appendix shows that the corresponding local state error satisfies
\begin{equation}
    \|e_{\tau_b}^{(b)}\|
    \lesssim
    \frac{1}{2}\,\kappa_b\,\tau_b^2\,\Delta t^2.
\end{equation}
This result directly exposes the desired behavior: blocks with low curvature can tolerate longer reuse intervals, while high-curvature blocks should be updated more often.

We formalize interval allocation as a budget-constrained optimization problem. Let $\rho$ denote the target speedup and $\tau_b$ the update interval for block $b$. We parameterize the block error as
\begin{equation}
    \ell_b(\tau_b) \propto \kappa_b^\alpha \tau_b^\beta,
\end{equation}
and minimize the total error under an average update-rate constraint:
\begin{equation}
    \min_{\tau_b \ge 1}\; \sum_{b=1}^{B} \kappa_b^\alpha \tau_b^\beta
    \qquad
    \text{s.t.}\quad
    \frac{1}{B}\sum_{b=1}^{B}\frac{1}{\tau_b} = \frac{1}{\rho}.
\end{equation}
The resulting closed-form optimum satisfies
\begin{equation}
    \tau_b^\star \propto \kappa_b^{-\alpha/(\beta+1)}.
\end{equation}
Using the theoretical default $(\alpha,\beta)=(2,4)$ yields
\begin{equation}
    \tau_b^\star \propto \kappa_b^{-2/5}.
\end{equation}
Thus, \cubic naturally assigns shorter intervals to high-curvature blocks and longer intervals to low-curvature ones.

In implementation, the continuous solution is further discretized, clipped to valid integer bounds, and re-normalized so that the achieved update rate remains close to the user-specified acceleration budget $\rho$.

\subsection{Selective Token DiT Forward}
\label{sec:runtime}

Given the partition $\mathcal{P}$ and the block intervals $\{\tau_b\}$, the active blocks at selective step $k$ are defined as
\begin{equation}
    \mathcal{A}_k = \{ b \in \mathcal{P} \mid s_k \bmod \tau_b = 0 \},
\end{equation}
where $s_k$ is the local step index within the selective phase. This induces the active and frozen token sets
\begin{equation}
    \Omega_k^{\mathrm{act}} = \bigcup_{b \in \mathcal{A}_k} B_b,
    \qquad
    \Omega_k^{\mathrm{frz}} = \Omega \setminus \Omega_k^{\mathrm{act}}.
\end{equation}

At each Transformer layer, \cubic recomputes the Query, Key, Value, and FFN only for active tokens. Frozen tokens reuse cached per-layer states and final outputs. For active tokens, attention is evaluated against the full sequence by combining newly computed Key/Value tensors for active tokens with cached Key/Value tensors for frozen tokens:
\begin{equation}
    Y_\ell^{\mathrm{act}} =
    \mathrm{Attn}\!\left(
        Q_\ell^{\mathrm{act}},
        K_\ell^{\mathrm{full}},
        V_\ell^{\mathrm{full}}
    \right).
\end{equation}
The final output of the selective step is then formed by merging newly computed active-token outputs with cached frozen-token outputs:
\begin{equation}
    v_k(\Omega) =
    \mathrm{Merge}\!\left(
        v_k^{\mathrm{act}}(\Omega_k^{\mathrm{act}}),
        v_{k-1,\mathrm{cache}}(\Omega_k^{\mathrm{frz}})
    \right).
\end{equation}
This selective forward engine is the component that actually yields computational savings: active tokens traverse the full DiT computation path, whereas frozen tokens participate only through cache reuse.

To ensure robustness, \cubic executes selective forward only when the required caches are already initialized and shape-consistent. Otherwise, the runtime automatically falls back to a full forward pass to refresh the cache state.

\subsection{End-to-End Inference Procedure}
\label{sec:end2end}

Putting the components together, \cubic performs inference as follows:
\begin{enumerate}
    \item In the warmup phase, it executes full forward passes to bootstrap both curvature observations and cache states.
    \item At each anchor step in the cruise phase, it updates the token-space curvature proxy and re-computes both the cache granularity and the block update intervals.
    \item At the subsequent selective steps, it recomputes only active tokens and reuses cached outputs for frozen tokens.
    \item In the cooldown phase, it returns to full forward passes to preserve final-generation fidelity.
\end{enumerate}

This pipeline turns local trajectory geometry into a concrete execution policy: the monitor estimates curvature, the partitioner designs granularity, the optimizer assigns refresh intervals, and the runtime engine realizes those decisions as selective token computation.
