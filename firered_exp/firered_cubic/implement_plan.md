 # cubic 实现 plan

先做 monitor-only 版本，不追求加速。
新增独立入口和独立 pipeline，先把 firered 的 phase 划分、anchor curvature、partition、tau 计划、可视化打通，确认 token grid 和 monitor 没理解错。

再做最小可行 selective。
只对“主生成图 token”做 selective；条件图 token 先全部设为 always-active。
这会少吃一点理论 speedup，但风险最低，因为条件图 token 虽然不参与最终输出切片，却强参与 attention 上下文。

selective engine 直接学 flux 的结构，不学 wan 的 runtime 形态。
我会更倾向于新建：

firered_exp/cubic_generate.py
firered_exp/firered_cubic/pipeline.py
firered_exp/firered_cubic/cubic_selective_forward.py
anchor monitor 只看主生成图。
noise_pred 最终会被切成 [:, :latents.size(1)] 再送入 scheduler，所以 curvature 也应该基于这部分主图 latent/noise 来算，不要把条件图 token 混进去。pipeline.py (line 819) pipeline.py (line 843)

CFG 分支缓存分开。
cond 和 uncond 两条分支都各自维护 output cache / KV cache，step plan 可以共享，但 cache key 不能共享。

## 最可能的坑

最大坑是 token 边界。
firered 的 image stream 里同时含主图 token 和条件图 token，step plan 只能作用在主图前缀，不能直接把整段 image sequence 当成一个平面做 block mask。

第二个坑是 attention processor 结构。
firered 是单一 QwenImageTransformerBlock 的双流 joint attention，没有 flux 那种 joint block + single block 两段式结构，所以 selective patch 点要改写它自己的 attention processor，而不是直接照搬 flux。transformer.py (line 578) transformer.py (line 940)

第三个坑是条件图 token 是否可冻结。
它们输入本身是静态的，但经过 timestep modulation 和 joint attention 后，内部表示未必能安全跨步复用。第一版最好别冻它们。

第四个坑是 CFG 后处理。
firered 在 neg + scale * (cond-neg) 后还做了一次 norm matching；anchor monitor 应该基于这个最终 noise_pred，否则 scheduler 真正用的速度场和 cubic 观察到的不一致。pipeline.py (line 835)

第五个坑是 cache_context("cond"/"uncond")。
这说明模型本身已经有分支语义；我们自己的 cubic cache 最好顺着这个分支语义走，而不是另起一套混在一起。

## 推荐落地顺序

我建议别一上来就做完整 selective sparse attention，而是：

CubicFireRedPipeline 全量 forward 版
curvature/partition/tau 可视化确认版
主图 selective + 条件图 always-active 版
