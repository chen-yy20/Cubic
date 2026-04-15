# 2026-04-15（flux2k partition 可视化改为显示 cooldown c）
- 日期：2026-04-15
- 目标模块：`flexflow/visualizer.py`、`flux2k_exp/flux2k_cubic/pipeline.py`
- 变更摘要：
  - 扩展 `CubicVisualizer/PartitionVisualizer` 接口，新增 `block_value_label`（默认 `tau`）。
  - summary 导出支持按标签输出统计：
    - 例如 `c_stats=...`；
    - `block_detail` 中按 `c=...` 展示每个 partition 的分配值。
  - frame strip 覆盖文字与色条标题由固定 `tau` 改为动态标签（flux2k下为 `c`）。
  - flux2k pipeline 可视化调用改为传入真实 `cooldown_by_block`，并设置 `block_value_label='c'`。
- 触达文件：
  - `flexflow/visualizer.py`
  - `flux2k_exp/flux2k_cubic/pipeline.py`
  - `worklog/cubic_worklog.md`

# 2026-04-15（flux2k optimizer增强：显式分配cooldown=0桶）
- 日期：2026-04-15
- 目标模块：`flux2k_exp/flux2k_cubic/optimizer.py`
- 变更摘要：
  - 针对“只出现 cooldown=1/2、0难以分配”的问题，调整 cooldown 分配策略：
    - 从“按绝对曲率缩放到整数”改为“按曲率排序分桶（rank bucketization）”；
    - 低曲率块优先落入 `cooldown=0` 桶，高曲率块分配到更大的 cooldown。
  - 增加保障：当 block 数>=2 且未出现 `cooldown=0` 时，强制将最小曲率块置为 0。
  - 目标是在短步数（尤其4步推理）中最大化相对性能收益，显式利用 `00/01/11` 三档。
- 触达文件：
  - `flux2k_exp/flux2k_cubic/optimizer.py`
  - `worklog/cubic_worklog.md`
- 验证命令：
  - `source ./.venv/bin/activate`
  - `python launch.py --model flux2k --method cubic`
- 验证结果：
  - 日志确认出现 `cooldown=[0,2]`；
  - selective 两步均生效，step2/step3 均保持显著低于anchor步的耗时。

# 2026-04-15（flux2k optimizer语义修正：每partition仅决定cooldown）
- 日期：2026-04-15
- 目标模块：`flux2k_exp/flux2k_cubic/optimizer.py`、`flux2k_exp/flux2k_cubic/pipeline.py`
- 变更摘要：
  - 按用户澄清，将 flux2k optimizer 从“策略标签+全局cooldown阶段”改为“token视角的 per-partition cooldown 分配”。
  - 新语义：
    - selective 总步数 `S = num_steps - anchor_steps`；
    - 每个 block 分配 `c_b ∈ [0, S]`；
    - 调度规则：冻结前 `S-c_b` 步，仅在最后 `c_b` 步 active。
  - 当 `S=2` 时自然对应 `00/01/11` 三种模式（由 `c_b=0/1/2` 决定）。
  - pipeline 侧移除旧的全局cooldown阶段分支，统一在 selective 阶段按 `cooldown_by_block` 构建 `StepPlan`。
  - 可视化复用接口保留：将 `cooldown_by_block` 映射为近似 interval 仅用于展示，不改变真实调度。
- 触达文件：
  - `flux2k_exp/flux2k_cubic/optimizer.py`
  - `flux2k_exp/flux2k_cubic/pipeline.py`
  - `worklog/cubic_worklog.md`
- 验证命令：
  - `source ./.venv/bin/activate`
  - `python launch.py --model flux2k --method cubic`
- 验证结果：
  - 运行通过；日志显示 `selective_steps=2` 且按 block cooldown 分配调度。
  - 示例：step2 active_blocks 约为 4/16 或 8/16，step3 active_blocks 16/16，符合“最后 cooldown 步 active”的规则。

# 2026-04-15（flux2k cubic：修复step2未加速，selective索引从1开始）
- 日期：2026-04-15
- 目标模块：`flux2k_exp/flux2k_cubic/pipeline.py`
- 变更摘要：
  - 修复 selective 相位首步（step2）经常退化为全量计算的问题。
  - 原因：`selective_step_idx` 从 0 开始时，`idx % tau == 0` 会使首个 selective 步大量 block 同时 active，导致 `full_compute=True`。
  - 调整：将 selective 相位索引改为从 1 开始：
    - `selective_step_idx = max(1, i - anchor_steps + 1)`
  - 结果：step2 与 step3 均进入 selective 计算形态。
- 触达文件：
  - `flux2k_exp/flux2k_cubic/pipeline.py`
  - `worklog/cubic_worklog.md`
- 验证命令：
  - `source ./.venv/bin/activate`
  - `python launch.py --model flux2k --method cubic`
- 验证结果：
  - 日志显示 step2/step3 均为 `full_compute=False`；
  - 计时上 step2 与 step3 均出现加速（例如 step2 ~65ms，step3 ~282ms，均低于anchor步）。

# 2026-04-15（flux2k cubic：适配visualize开关）
- 日期：2026-04-15
- 目标模块：`flux2k_exp/cubic_generate.py`
- 变更摘要：
  - 新增参数 `--visualize_curvature`，用于显式控制 cubic 可视化导出。
  - 开关关闭时：
    - `run_straightness_sampling` 不传 `cubic_results_dir`（避免分区/频率可视化开销）。
    - 不导出 `straightness_vis` 与 `straightness_summary.txt`。
  - 开关开启时：保持原有可视化导出行为。
  - 将 `visualize_curvature` 写入 params_metrics，便于复现实验配置。
- 触达文件：
  - `flux2k_exp/cubic_generate.py`
  - `worklog/cubic_worklog.md`

# 2026-04-15（flux2k ori：对齐多seed生成与warmup逻辑）
- 日期：2026-04-15
- 目标模块：`flux2k_exp/ori_generate.py`
- 变更摘要：
  - 为 flux2k ori 入口增加多seed生成流程，与 flux1 风格保持一致：
    - 默认 `num_images=3`，seed 序列为 `[base_seed, base_seed+1, base_seed+2]`。
  - 增加 warmup 预热流程：
    - 默认 `warmup_runs=2`，在正式计时前执行两次预热采样。
  - 正式阶段按每个 seed 输出单独图片：`..._seed{seed}.png`，避免覆盖。
  - 参数落盘改为多图结构：`output_images/base_seed/seeds/num_images/warmup_runs`。
- 计时策略：
  - 保持原有计时口径不变：`load_pipeline`、`inference`、`save_image`。
  - warmup 不包裹在 `get_timer("inference")` 中，不计入正式计时统计。
- 触达文件：
  - `flux2k_exp/ori_generate.py`
  - `worklog/cubic_worklog.md`

# 2026-04-15（flux2k：warmup两次后按3个seed生成3图，对齐flux1）
- 日期：2026-04-15
- 目标模块：`flux2k_exp/cubic_generate.py`、`flux2k_exp/flux2k_cubic/pipeline.py`
- 变更摘要：
  - 入口改为多seed生成：默认 `num_images=3`，基于 `base_seed` 生成 `[seed, seed+1, seed+2]`。
  - 增加 warmup 流程：正式计时前执行 `warmup_runs=2` 次预热采样。
  - 为保证计时精度，warmup 调用不记录 step 计时：在 `run_straightness_sampling` 增加 `enable_step_timing` 开关。
  - 正式阶段按每个 seed 输出一张图：
    - `..._seed42.png`
    - `..._seed43.png`
    - `..._seed44.png`
  - 每个seed的straightness可视化与summary分目录保存到 `seed{seed}/`，避免覆盖。
  - 质量评估逻辑适配多seed：按seed记录并汇总 `mean`（当 `--quality_eval` 启用时）。
- 计时结果约束：
  - 保持仅4类计时：`load_pipeline`、`inference`、`step_xxxx_total`、`save_image`。
  - 验证中样本数符合预期：`inference`/`save_image`/`step_xxxx_total` 均为 3 次样本（仅正式3次生成）。
- 触达文件：
  - `flux2k_exp/cubic_generate.py`
  - `flux2k_exp/flux2k_cubic/pipeline.py`
  - `worklog/cubic_worklog.md`
- 验证命令：
  - `source ./.venv/bin/activate`
  - `python launch.py --model flux2k --method cubic`
- 验证结果：
  - 运行通过；先warmup 2次，再生成seed42/43/44三张图；计时统计结构与样本数符合预期。

# 2026-04-15（flux2k：计时项收敛，仅保留4类）
- 日期：2026-04-15
- 目标模块：`flux2k_exp/flux2k_cubic/pipeline.py`、`flux2k_exp/cubic_generate.py`
- 变更摘要：
  - 按要求收敛计时粒度，仅保留以下4类：
    1) `load_pipeline`
    2) `inference`
    3) `step_xxxx_total`
    4) `save_image`
  - 移除 step 内所有细分子计时（build_plan/cond_forward/uncond_forward/unpack/anchor_monitor/partition_opt/visualize/straightness/scheduler_step）。
  - 移除 `quality_eval` 的独立计时项，避免新增第5类计时。
- 变更依据：
  - 入口计时对齐 ori 的主阶段统计；
  - 逐步统计只保留单一总耗时，便于快速对比各步开销。
- 触达文件：
  - `flux2k_exp/flux2k_cubic/pipeline.py`
  - `flux2k_exp/cubic_generate.py`
  - `worklog/cubic_worklog.md`

# 2026-04-15（flux2k：Selective Forward 首版落地，cond/uncond 统一策略）
- 日期：2026-04-15
- 目标模块：`flux2k_exp/flux2k_cubic/cubic_selective_forward.py`、`flux2k_exp/flux2k_cubic/pipeline.py`
- 变更摘要：
  - 实现 Selective Forward 首版：
    - 在 selective 步仅对 `StepPlan.active_token_indices` 对应 token 执行 transformer forward；
    - 对 frozen token 复用上一步输出缓存（velocity cache 风格）；
    - anchor/full 步自动刷新缓存。
  - 明确本轮边界：先不引入 KV cache sparse attention，仅实现 token 级 selective 执行。
  - 按要求统一 cond/uncond 路径策略：
    - 两条路径在 selective 阶段使用同一个 `step_plan`；
    - 两条路径都走 `Flux2KSelectiveForwardEngine.forward(...)`；
    - 各自使用独立 cache key（`flux2k_cond` / `flux2k_uncond`）避免互相污染。
- 变更依据：
  - 对齐增量实现顺序：先打通 partition/optimizer，再落 selective forward。
  - 与 `implement.md` 的 selective 调度意图一致，先完成 active/frozen token 执行分离，再进入 KV cache 细化。
- 触达文件：
  - `flux2k_exp/flux2k_cubic/cubic_selective_forward.py`
  - `flux2k_exp/flux2k_cubic/pipeline.py`
  - `worklog/cubic_worklog.md`
- 验证命令：
  - `source ./.venv/bin/activate`
  - `python launch.py --model flux2k --method cubic`
- 验证结果：
  - 运行通过，日志显示 selective 计划生效（`step=3 active_tokens=512/4096 full_compute=False`）。
  - cond/uncond 路径均已应用相同 selective 策略并成功完成全流程。

# 2026-04-15（flux2k：先对接 partitioner/optimizer，再进入 selective 执行）
- 日期：2026-04-15
- 目标模块：`flux2k_exp/flux2k_cubic/pipeline.py`
- 变更摘要：
  - 在 `run_straightness_sampling` 中接入 `CurvatureMonitor`、`AdaptivePartitioner`、`UpdateFrequencyOptimizer`、`SelectiveStepScheduler`、`CubicVisualizer`。
  - 采样阶段按“前两步 anchor、后两步 selective”执行：
    - anchor 步进行 full forward + curvature 更新；
    - 第二个 anchor 产出曲率图后立即执行 partition 与 tau 分配；
    - selective 步生成 `StepPlan` 并输出 active/frozen token 统计。
  - 本轮按“先打通调度链路”的目标，selective 步暂保持 full forward，不提前引入稀疏执行回归风险。
- 变更依据：
  - 对齐 `flexflow/docs/implement.md` 2.2、2.4、2.5：anchor 负责全量监测、统一分块、tau 优化分配。
  - 对齐 `flexflow/docs/theory.md` 的曲率驱动调度思想：先稳定建立 curvature->partition->frequency 映射，再下沉 selective 计算与缓存复用。
- 触达文件：
  - `flux2k_exp/flux2k_cubic/pipeline.py`
  - `flux2k_exp/cubic_generate.py`
  - `worklog/cubic_worklog.md`
- 验证命令：
  - `source ./.venv/bin/activate`
  - `python launch.py --model flux2k --method cubic`
- 验证结果：
  - 运行通过；日志中可见：
    - `Cubic anchor plan: step=1 blocks=16 speedup=1.959 tau=[1,4]`
    - `Cubic selective plan: step=2 ... full_compute=True`
    - `Cubic selective plan: step=3 active_tokens=512/4096 full_compute=False`
  - 图像、straightness 可视化与 summary 均正常导出。

# 2026-04-15 flux2k cubic anchor/selective分段与selective forward主流程

## 目标
实现flux2k cubic推理流程的anchor/selective分段，前两步anchor（全量forward+monitor），后两步selective（仅active token forward，frozen token复用缓存），并完成主流程可跑通。

## 主要变更
- 新增 flux2k_exp/flux2k_cubic/cubic_selective_forward.py，参考flux1 cubic selective forward实现，支持KV cache与active token selective forward。
- pipeline.py中引入cubic_selective_engine，并在run_straightness_sampling中实现anchor/selective分段，selective步调用engine.forward。
- selective步暂用“前一半token为active”策略，后续可接入partition/step_plan。

## 验证
- 运行 `source ./.venv/bin/activate && python launch.py --model flux2k --method cubic`，anchor步与selective步均可跑通，输出straightness记录、图片与可视化。
- selective步未报错，主流程通过。
- 期间遇到一次CUDA OOM，重试后正常。

## 后续建议
- 进一步完善step_plan与partition集成，实现真正的blockwise selective。
- 支持frozen token velocity cache复用。
- 完善与flexflow/partition、optimizer等模块的对接。
# Cubic 工作日志

## 2026-04-15（flux2k 入口重构：cubic 逻辑迁移到 flux2k_cubic 模块）
- 目标模块：
  - `flux2k_exp/flux2k_cubic/pipeline.py`
  - `flux2k_exp/flux2k_cubic/__init__.py`
  - `flux2k_exp/cubic_generate.py`
- 变更摘要：
  - 将 flux2k cubic 的核心执行逻辑从入口脚本迁移到 `flux2k_cubic` 模块：
    - `load_pipeline(...)`
    - `run_straightness_sampling(...)`
    - `save_straightness_summary(...)`
    - `resolve_output_paths(...)`
  - `flux2k_exp/cubic_generate.py` 现在仅保留参数解析、初始化、调用模块、结果保存，角色与 `ori_generate.py` 对齐。
  - 修复迁移时引入的函数名回归（`_resolve_output_paths` -> `resolve_output_paths`）。
- 变更依据：
  - 对齐 `flexflow/docs/implement.md` 的解耦入口原则（通过新增/独立模块接入 cubic，避免在入口脚本堆积实现细节）。
  - 保持 origin runtime 不变，符合“入口脚本负责启动、实现放在模块中”的结构约束。
- 受影响文件：
  - `flux2k_exp/flux2k_cubic/pipeline.py`（新增）
  - `flux2k_exp/flux2k_cubic/__init__.py`（新增）
  - `flux2k_exp/cubic_generate.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`source ./.venv/bin/activate && python launch.py --model flux2k --method cubic`
  - 结果：通过。4 步采样与 straightness 可视化正常，产物路径保持不变。

## 2026-04-15（flux2-klein：4步 straightness 可视化接入）
- 目标模块：
  - `flux2k_exp/cubic_generate.py`
  - `launch.py`
- 变更摘要：
  - 新增 `flux2k_exp/cubic_generate.py` 的 cubic 入口实现，保持 origin runtime 不改动，独立执行 flux2-klein 采样并接入 straightness 评估链路。
  - 在 4 步去噪循环中按 theory 的 endpoint drift 关系记录 straightness 指标，导出：
    - `straightness_vis/` 可视化图（含 overview 与热力图）
    - `straightness_summary.txt`（每步 `mean_ratio/corr` 摘要）
  - 在 `launch.py` 为 `flux2k` 增加 `cubic` 路由，支持统一命令：`python launch.py --model flux2k --method cubic`。
  - 修复首轮运行中的显存累积问题：为 cubic 入口补充 `torch.inference_mode()`，避免推理图构建导致 OOM。
- 变更依据：
  - 对齐 `flexflow/docs/theory.md` §4：使用终点预测漂移与局部曲率比例关系进行 straightness 观测。
  - 对齐 `flexflow/docs/implement.md` 3.2（解耦入口）与“保留 origin runtime”原则：通过新增入口模块并行接入，不改 `flux2k_exp/flux2_klein/` 原始推理实现。
- 受影响文件：
  - `flux2k_exp/cubic_generate.py`
  - `launch.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`source ./.venv/bin/activate && python launch.py --model flux2k --method cubic`
  - 结果：通过。4 步采样执行完成，控制台输出 3 条 straightness 记录（step1-3），并成功写出：
    - `results/flux2/flux2-klein/cubic/cat_holding_sign_that/straightness_vis/`
    - `results/flux2/flux2-klein/cubic/cat_holding_sign_that/straightness_summary.txt`
    - `results/flux2/flux2-klein/cubic/cat_holding_sign_that/cubic_params_metrics.json`

## 2026-04-15（flux selective attention 修正：encoder 全量保留，仅 image 做 selection+KV cache）
- 目标模块：
  - `flux1_exp/flux_cubic/cubic_selective_forward.py`
- 变更摘要：
  - 在 Flux selective attention 中明确划分 token 角色：
    - `encoder hidden states` 全量参与序列拼接与注意力计算，不参与 selection，不进入 KV cache 的更新索引。
    - `image hidden states` 才按 `active_token_indices` 走 selective 计算，并按 layer 维护/更新 image KV cache。
  - 将 single block 中基于 runtime 的前缀长度字段从 `text_tokens` 收敛为 `encoder_tokens`，并加上边界校验，避免误把 encoder 纳入 image selection 逻辑。
  - 修复 Flux rotary embedding 切片错误：`image_rotary_emb` 在 diffusers 中为 `(cos, sin)` 二元组，新增 `_slice_rotary_emb(...)` 按序列维切分后再传给 `apply_rotary_emb`，避免 tuple 直接切片导致的空频率输入。
- 变更依据：
  - 对齐 `flexflow/docs/implement.md` 2.6：选择性稀疏前向仅优化 image token，条件信息（encoder/context）应保持全量参与，确保条件一致性。
  - 保持 origin 接口不变，仅在 `flux_cubic` 解耦模块内部修正 selective+cache 路径。
- 受影响文件：
  - `flux1_exp/flux_cubic/cubic_selective_forward.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`source ./.venv/bin/activate && python launch.py --model flux --method cubic`
  - 结果：通过。3 个 seed 端到端运行完成，生成与质量评测流程正常，计时统计中 `selective_forward/full_forward` 均正常出现。

## 2026-04-14（README 同步 Flux1 cubic 能力与使用说明）
- 目标模块：
  - `README.md`
- 变更摘要：
  - 更新仓库总览，明确支持 `flux` 的 `ori/cubic` 启动映射。
  - 补充 Flux1 origin 启动示例与 Flux1 cubic 运行入口。
  - 在 cubic 章节补充 Flux1 已实现能力：三阶段流程、anchor 监测分块优化、phase3 selective forward、可视化导出、质量评测汇总。
  - 在输出章节补充 Flux1 cubic 目录结构与关键产物说明（图片、params_metrics、curvature/partition 可视化）。
  - 将质量指标章节从 WAN 扩展为 WAN/Flux，并说明 Flux 图像场景下 `FVD` 可能为空。
- 变更依据：
  - 对齐当天已完成的 Flux1 cubic 实装进展，避免 README 与实际可运行功能不一致。
- 受影响文件：
  - `README.md`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`source ./.venv/bin/activate && python launch.py --model flux --method cubic`
  - 结果：通过（本次文档更新未引入代码路径变更）。

## 2026-04-14（flux cubic 接入 quality eval）
- 目标模块：
  - `flux1_exp/cubic_generate.py`
- 变更摘要：
  - 参照 WAN 的入口逻辑，在 Flux cubic 生成后接入 `evaluate_quality_with_origin(...)`。
  - 按 seed 对每张生成图执行图像质量评测，并将结果写入 `per_seed`。
  - 对 `psnr/ssim/lpips/fid/fvd` 计算跨 seed 均值，写入 `mean`。
  - 将 `quality_metrics` 一并保存到 `cubic_params_metrics.json`，并在参数中记录 `quality_eval` 开关。
- 变更依据：
  - 对齐 WAN 的质量评测闭环：生成产物后立即与 `ori` 基线做同 prompt/seed 对比，并统一回写到 params+metrics 文件。
  - Flux 为多 seed 图像输出，采用“逐图评测 + 汇总均值”更便于横向比较。
- 受影响文件：
  - `flux1_exp/cubic_generate.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`source ./.venv/bin/activate && python -m py_compile flux1_exp/cubic_generate.py`
  - 结果：通过。
  - 命令：`source ./.venv/bin/activate && python launch.py --model flux --method cubic`
  - 结果：通过，3 seed 生成完成并完成质量评测，`cubic_params_metrics.json` 已包含 `quality_metrics`。

## 2026-04-14（flux selective forward 首版接入）
- 目标模块：
  - `flux1_exp/flux_cubic/cubic_selective_forward.py`
  - `flux1_exp/flux_cubic/pipeline.py`
- 变更摘要：
  - 新增 Flux 侧 `FluxSelectiveForwardEngine`，在 `phase3_cruise_selective` 中执行 active token 前向，并使用上一轮缓存的 velocity 对 frozen token 回填。
  - 在 `CubicFluxPipeline` 中接入 `SelectiveStepScheduler` 与 selective engine：
    - anchor/warmup/cooldown 仍走 full forward 并刷新缓存；
    - cruise 非 anchor 根据 `partition + tau` 生成 `StepPlan`，驱动 selective forward。
  - 针对 Flux 与 WAN 映射差异，增加了 `hidden_states` token 序列与 `img_ids` 的一致性校验（长度/索引范围），异常时自动回退 full forward，避免 token 错位更新。
- 变更依据：
  - 对齐 `flexflow/docs/implement.md` 中 2.2（anchor 刷新）与 2.6（cruise 非 anchor selective）的流程约束。
  - 参考 `wan_exp/wan_cubic/cubic_selective_forward.py` 的缓存式接口组织，同时按 Flux 的 `[B,L,C]` packed latent 序列语义适配 token 索引。
- 受影响文件：
  - `flux1_exp/flux_cubic/cubic_selective_forward.py`（新增）
  - `flux1_exp/flux_cubic/pipeline.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`source ./.venv/bin/activate && python -m py_compile flux1_exp/flux_cubic/cubic_selective_forward.py flux1_exp/flux_cubic/pipeline.py`
  - 结果：通过（无语法错误）
  - 命令：`source ./.venv/bin/activate && python launch.py --model flux --method cubic`
  - 结果：通过，三 seed 端到端完成，计时中可见 `selective_forward` 栈（93 次调用），流程可执行。

## 2026-04-14（flux step31 过细分块修正：加入 block token 上下限）
- 目标模块：
  - `flexflow/partition.py`
  - `flux1_exp/flux_cubic/pipeline.py`
- 变更摘要：
  - 在 `AdaptivePartitioner` 中新增候选筛选约束：`min_block_tokens` / `max_block_tokens`。
  - block size 搜索时，先计算所有候选指标，再优先在 token 数满足区间的候选中选最优；若区间内无候选，再回退到全候选，保证鲁棒性。
  - 在 Flux cubic pipeline 中设置并同步：`min_block_tokens=16`、`max_block_tokens=256`，避免出现 `(1,2,2)` 这类过细分块。
- 变更依据：
  - `step_0031_partition_summary_seed43.txt` 显示旧策略在仅用 CV 一致性选型时会偏向过细块（`selected_block_size=(1,2,2)`，1024 blocks），与“不过粗不过细”的目标不符。
  - 对齐“全局 blocksize 选择”目标，在一致性之外加入显式尺度约束。
- 受影响文件：
  - `flexflow/partition.py`
  - `flux1_exp/flux_cubic/pipeline.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`source ./.venv/bin/activate && python -m py_compile flexflow/partition.py flux1_exp/flux_cubic/pipeline.py`
  - 结果：通过（无输出，退出码 0）
  - 命令：`source ./.venv/bin/activate && python launch.py --model flux --method cubic`
  - 结果：通过。
  - 观测：`step_0031_partition_summary_seed43.txt` 的 `selected_block_size` 由 `(1,2,2)` 变为 `(1,8,2)`；`step_0031_partition_summary_seed42.txt` 为 `(1,4,4)`，block tokens 均为 16，未再出现过细 4-token 分块。

## 2026-04-14（flux tau 区分度修正：曲率对比度增益 + 配置同步）
- 目标模块：
  - `flexflow/optimizer.py`
  - `flux1_exp/flux_cubic/pipeline.py`
- 变更摘要：
  - 在 `UpdateFrequencyOptimizer` 中新增 `curvature_contrast_gamma`，在 `tau` 映射前先做曲率对比度放大：
    - `kappa_scaled = (kappa / median(kappa)) ** gamma`
    - 再执行 `tau ∝ kappa_scaled^{-alpha/(beta+1)}`。
  - 在 `CubicFluxPipeline` 中补充 `_sync_cubic_modules_config()`，确保运行时参数（`target_speedup/alpha/beta/tau_range/blocksize`）真实同步到 partitioner/optimizer，而非只改 pipeline 字段。
  - 为 flux 默认启用较强的曲率区分增益：`curvature_contrast_gamma = 1.8`。
- 变更依据：
  - 结合 `partition_summary` 观察：在粗分块阶段（如 `(1,16,16)`）曲率差异存在但共享映射灵敏度不足，离散后大量塌缩到 `tau=2`。
  - 该修正保持算法框架不变，只增强曲率到 `tau` 的单调映射对比度，适配 flux 图像场景的曲率分布特征。
- 受影响文件：
  - `flexflow/optimizer.py`
  - `flux1_exp/flux_cubic/pipeline.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`source ./.venv/bin/activate && python -m py_compile flexflow/optimizer.py flux1_exp/flux_cubic/pipeline.py`
  - 结果：通过（无输出，退出码 0）
  - 命令：`source ./.venv/bin/activate && python launch.py --model flux --method cubic`
  - 结果：通过，`anchor_step_partitioner/anchor_step_optimizer` 计时正常。
  - 观测：`step_0007_partition_summary_seed42.txt` 中 `interval_stats` 由先前几乎全 `tau=2` 变为 `min=1, max=3, mean=2.19`，分化明显提升。

## 2026-04-14（flux1 第二阶段：anchor 分块与 tau 分配）
- 目标模块：
  - `flux1_exp/flux_cubic/pipeline.py`
  - `flexflow/visualizer.py`
  - `flux1_exp/cubic_generate.py`
- 变更摘要：
  - 在 Flux cubic 的 anchor 步中接入 `AdaptivePartitioner` 与 `UpdateFrequencyOptimizer`。
  - 监测链路更新为：`monitor.update -> partitioner.partition -> monitor.get_block_curvatures -> optimizer.optimize`。
  - 新增分块与 `tau` 可视化导出调用：`visualize_partition_and_frequency(...)`。
  - 为分块可视化新增 `file_suffix` 支持，避免多 seed 下 `partition_vis` 文件相互覆盖。
  - 在 flux 入口中传递 `cubic_target_speedup=SPEEDUP`，用于 `tau` 分配目标约束。
- 变更依据：
  - 对齐 [flexflow/docs/implement.md](../flexflow/docs/implement.md) 的 2.4（全局 blocksize + 均匀切分）与 2.5（`tau_b` 优化分配）。
  - 保持第一阶段策略：先在 anchor 步完成观测与调度，selective forward 仍后续迭代接入。
- 受影响文件：
  - `flux1_exp/flux_cubic/pipeline.py`
  - `flexflow/visualizer.py`
  - `flux1_exp/cubic_generate.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`source ./.venv/bin/activate && python -m py_compile flux1_exp/flux_cubic/pipeline.py flexflow/visualizer.py flux1_exp/cubic_generate.py`
  - 结果：通过（无输出，退出码 0）
  - 命令：`source ./.venv/bin/activate && python launch.py --model flux --method cubic`
  - 结果：通过，计时统计中出现 `anchor_step_partitioner` 与 `anchor_step_optimizer`。
  - 说明：当前 `flux1_exp/cubic_generate.py` 中 `curvature_visualize=False` 时不会导出可视化文件；设为 `True` 即会输出曲率与分块/tau 可视化。

## 2026-04-14（flux1 curvature 可视化防覆写 + 单行排版）
- 目标模块：
  - `flexflow/visualizer.py`
  - `flux1_exp/flux_cubic/pipeline.py`
  - `flux1_exp/cubic_generate.py`
- 变更摘要：
  - 为曲率总览导出新增 `file_suffix` 参数，支持按运行标签输出不同文件名（例如 `curvature_overview_seed42.png`）。
  - 调整 `curvature_overview` 排版为固定单行多列（1 行，anchor steps 按列展示）。
  - 在 flux cubic 调用链中新增 `cubic_visual_tag`，并按 seed 传递，确保 3 个 seed 产生 3 张曲率总览图。
  - 保留历史可视化文件，不再在每次导出曲率时清空目录，避免多 seed 运行互相覆盖。
- 变更依据：
  - 满足“多 seed 生成应得到多张 curvature overview”与“overview 改为单行排版”的可视化需求。
  - 不改动 monitor 曲率计算逻辑，仅调整可视化导出策略与调用参数。
- 受影响文件：
  - `flexflow/visualizer.py`
  - `flux1_exp/flux_cubic/pipeline.py`
  - `flux1_exp/cubic_generate.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`source ./.venv/bin/activate && python -m py_compile flexflow/visualizer.py flux1_exp/flux_cubic/pipeline.py flux1_exp/cubic_generate.py`
  - 结果：通过（无输出，退出码 0）
  - 命令：`source ./.venv/bin/activate && python launch.py --model flux --method cubic`
  - 结果：通过，生成 `curvature_overview_seed42.png`、`curvature_overview_seed43.png`、`curvature_overview_seed44.png`。

## 2026-04-14（flux1 第一阶段：分阶段 pipeline + anchor monitor + 可视化）
- 目标模块：
  - `flux1_exp/flux_cubic/pipeline.py`
  - `flux1_exp/cubic_generate.py`
  - `launch.py`
  - `flux1_exp/flux_cubic/__init__.py`
- 变更摘要：
  - 新增 `CubicFluxPipeline`，在 Flux 图像生成路径中加入与 WAN 对齐的阶段判定逻辑：warmup / cruise(anchor/selective) / cooldown。
  - 在 warmup 尾步与 cruise anchor 步接入 `flexflow.CurvatureMonitor`，并将 Flux 的 packed latent/noise 还原为 5D 形状后再进行 monitor 更新。
  - 接入 `flexflow.CubicVisualizer` 导出曲率总览图（`curvature_vis/curvature_overview.png`）。
  - 新增 `flux1_exp/cubic_generate.py` 作为独立 cubic 入口脚本，保持 origin 入口不改动。
  - 在 `launch.py` 中新增 `flux` 模型族的 `cubic` 路由，支持 `python launch.py --model flux --method cubic`。
- 变更依据：
  - 对齐 [flexflow/docs/implement.md](../flexflow/docs/implement.md) 的 2.1（三阶段推理）与 2.2（anchor step）规范。
  - 对齐 4.4 monitor 约定：anchor 使用 endpoint drift 更新曲率，并在进入公式前将 timestep 归一化到 `[0, 1]`。
  - 按“保留 origin runtime”的要求，采用新增 `flux_cubic` 模块和独立入口脚本实现。
- 受影响文件：
  - `flux1_exp/flux_cubic/__init__.py`（新增）
  - `flux1_exp/flux_cubic/pipeline.py`（新增）
  - `flux1_exp/cubic_generate.py`（新增实现）
  - `launch.py`（新增 flux cubic 映射）
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`source ./.venv/bin/activate && python -m py_compile flux1_exp/flux_cubic/pipeline.py flux1_exp/cubic_generate.py launch.py`
  - 结果：通过（无输出，退出码 0）
  - 命令：`source ./.venv/bin/activate && python launch.py --model flux --method cubic`
  - 结果：通过，完成 3 张图片生成并输出计时统计；曲率可视化已写入 `results/flux/cubic/photorealistic_cute_cat,_wearing_2x/curvature_vis/`。

## 2026-04-14（flux1 baseline ori_generate 清理）
- 目标模块：
  - `flux1_exp/ori_generate.py`
- 变更摘要：
  - 删除了所有与 `jano` 相关的代码、环境变量和质量评估分支。
  - 将入口整理为纯 `ori` baseline：统一加载模型、执行 warmup、逐 seed 生成并落盘。
  - 将结果保存逻辑收敛为与 WAN origin 类似的 `params + timing_stats` 写盘方式，质量指标固定为 `None`。
- 变更依据：
  - 按照“保留 baseline、去掉前置加速实验逻辑”的要求，先把 flux1 的 origin 入口理顺，再进入后续 cubic 版本。
  - 落盘结构沿用 `results/flux/ori/{prompt_id}`，便于与其他 origin 入口保持一致。
- 受影响文件：
  - `flux1_exp/ori_generate.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`source ./.venv/bin/activate && python -m py_compile flux1_exp/ori_generate.py`
  - 结果：通过（无输出，退出码 0）

## 2026-04-08
- 目标模块：`wan_exp/wan/text2video.py`
- 变更摘要：
  - 为每个去噪步骤添加了 4 阶段推理分类：
    - `phase1_warmup`
    - `phase2_cruise_anchor`
    - `phase3_cruise_selective`
    - `phase4_cooldown`
  - 在主采样循环中增加了逐步阶段日志（仅 rank 0），同时记录 `step` 和 `phase`。
- 变更依据：
  - 遵循了 [flexflow/docs/implement.md](../flexflow/docs/implement.md) 中“2.1 三阶段推理”和“2.2 Anchor Step”的思路：warmup + cruise(anchor/selective) + cooldown 自然可映射为 4 个阶段。
  - 只增加阶段计算和日志，保持行为兼容，没有修改 scheduler/model forward API。
- 受影响文件：
  - `wan_exp/wan/text2video.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`python3 -m py_compile wan_exp/wan/text2video.py`
  - 结果：通过（无输出，退出码 0）

## 2026-04-08（后续）
- 目标模块：
  - `wan_exp/wan/image2video.py`
  - `wan_exp/wan/first_last_frame2video.py`
  - `wan_exp/wan/vace.py`
- 变更摘要：
  - 在上述所有 WAN 生成循环中加入了相同的 4 阶段分类与逐步阶段日志。
  - 对于 `WanVaceMP`，阶段日志仅在 `rank == 0` 时输出，以避免多进程重复日志。
- 变更依据：
  - 让 WAN 的文本/视频/图生视频/VACE 入口都保持一致的阶段可观测性。
  - 复用与 `WanT2V` 相同的阶段边界和 anchor 间隔，避免分支漂移。
- 验证命令与结果：
  - 命令：`python3 -m py_compile wan_exp/wan/text2video.py wan_exp/wan/image2video.py wan_exp/wan/first_last_frame2video.py wan_exp/wan/vace.py`
  - 结果：通过（无输出）

## 2026-04-08（通过新文件解耦）
- 目标模块：
  - `wan_exp/cubic_text2video.py`
  - `wan_exp/cubic_generate.py`
  - `.github/agents/cubic-origin-implementer.agent.md`
- 变更摘要：
  - 回退了原始 WAN 运行时代码中的阶段日志修改，保持原始行为不变。
  - 在 `wan_exp/cubic_text2video.py` 中新增了解耦的 Cubic T2V 实现。
  - 新增独立的 Cubic 入口脚本 `wan_exp/cubic_generate.py`（仅 T2V），可在不修改原始模块的情况下运行带阶段日志的生成流程。
  - 更新了 Cubic Origin Implementer agent 的约束：WAN 默认范围为 T2V，且实现应优先使用新增模块/文件。
- 变更依据：
  - 满足仓库中“保留原始代码，通过增量模块实现扩展”的要求。
  - 让 cubic 演化保持隔离、便于审查。
- 验证命令与结果：
  - 命令：`python3 -m py_compile wan_exp/wan/text2video.py wan_exp/wan/image2video.py wan_exp/wan/first_last_frame2video.py wan_exp/wan/vace.py wan_exp/cubic_text2video.py wan_exp/cubic_generate.py`
  - 结果：通过（无输出）

## 2026-04-08（anchor monitor initial）
- 目标模块：
  - `wan_exp/cubic/monitor.py`
- 变更摘要：
  - 为 anchor-step 使用新增了 `CurvatureMonitor` 的初始实现。
  - 实现了端点更新：`endpoint = x + (1 - t) * v`。
  - 实现了基于漂移的曲率估计，以及使用 `patch_size` 的 latent-to-token pooling。
  - 通过百分位数（`curvature_percentile`，默认 0.9）实现了 block 级曲率聚合。
  - 增加了 monitor 状态重置。
- 变更依据：
  - 遵循 [flexflow/docs/implement.md](../flexflow/docs/implement.md) 中 4.4 的 anchor-step monitor 行为和数据流描述。
  - 保持实现解耦在 `wan_exp/cubic/` 下，不修改原始 WAN runtime 模块。
- 受影响文件：
  - `wan_exp/cubic/__init__.py`
  - `wan_exp/cubic/monitor.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`python3 -m py_compile wan_exp/cubic/monitor.py`
  - 结果：通过（无输出）
  - 命令：`source ./.venv/bin/activate && python launch.py --model wan --method cubic`
  - 结果：失败，原因是环境依赖问题：`wan_exp/wan/modules/attention.py` 中 `FLASH_ATTN_2_AVAILABLE` 为 `false` 触发 `AssertionError`，与该 monitor 逻辑无关。

## 2026-04-08（monitor 移到 flexflow，完成集成）
- 目标模块：
  - `flexflow/monitor.py`
  - `wan_exp/cubic_text2video.py`
- 变更摘要：
  - 在 monitor 迁移后，将 `CubicWanT2V` 接入 `flexflow.CurvatureMonitor`。
  - 在最后一个 warmup 步骤和 cruise anchor 步骤加入 monitor 触发点。
  - 在调用 monitor update 前，将 timestep 归一化到 `[0, 1]`。
  - 在 anchor 更新时增加曲率图形状和均值日志。
- 变更依据：
  - 对齐 [flexflow/docs/implement.md](../flexflow/docs/implement.md) 中 4.4 的 anchor-step monitor 更新流程，同时保持 WAN 原始代码不变。
  - 确保在真实 WAN cubic 采样中可以观测 monitor，便于联合调试。
- 验证命令与结果：
  - 命令：`python3 -m py_compile flexflow/monitor.py wan_exp/cubic_text2video.py wan_exp/cubic_generate.py`
  - 结果：通过（无输出）
  - 命令：`source ./.venv/bin/activate && python launch.py --model wan --method cubic`
  - 结果：通过，端到端生成完成，输出保存到 `results/wan/1.3B/cubic/two_anthropomorphic_cats_comfy/cubic_two_anthropomorphic_cats_comfy_t2v-1.3B.mp4`。

## 2026-04-08（anchor curvature visualization export）
- 目标模块：
  - `flexflow/monitor.py`
  - `wan_exp/cubic_text2video.py`
  - `wan_exp/cubic_generate.py`
- 变更摘要：
  - 在 `CurvatureMonitor.update(..., step_idx=step_idx)` 中增加了 anchor 曲率历史记录。
  - 新增 `CurvatureMonitor.export_visualizations(output_dir)`，用于导出每个 anchor 的可视化产物：
    - `step_XXXX_map.npz`（原始曲率图）
    - `step_XXXX_mean_hw.png`（时间维均值热图）
    - `step_XXXX_time_strip.png`（选定时间切片条带）
    - `step_XXXX_meta.txt`（step/t/shape/stats）
  - 将导出调用接入 `CubicWanT2V.generate(..., results_dir=results_dir)`，并在 `cubic_generate.py` 中传入 `args.output_dir`。
- 变更依据：
  - 在相同结果目录下提供实用、低开销的 anchor 曲率图可视化，便于调试和对比。
  - 保持实现与原始 WAN runtime 模块解耦。
- 验证命令与结果：
  - 命令：`python3 -m py_compile flexflow/monitor.py wan_exp/cubic_text2video.py wan_exp/cubic_generate.py`
  - 结果：通过
  - 命令：`source ./.venv/bin/activate && python launch.py --model wan --method cubic`
  - 结果：通过，可视化文件生成在 `results/wan/1.3B/cubic/two_anthropomorphic_cats_comfy/curvature_vis`。

## 2026-04-08（仅保留 mean hw 总览图）
- 目标模块：
  - `flexflow/monitor.py`
- 变更摘要：
  - 将 anchor 曲率可视化导出收敛为默认仅输出一张 `curvature_overview.png`。
  - 将 `npz`、逐步 strip、meta 等重型输出改为显式开关，默认关闭。
  - 导出前清理 `curvature_vis` 子目录，避免残留旧的重型结果文件。
- 变更依据：
  - 满足“只需要生成一张图（mean hw 总览图）”的要求，同时保留重型导出能力但默认不启用。
  - 继续保持与原始 WAN 运行时代码解耦。
- 验证命令与结果：
  - 命令：`python3 -m py_compile flexflow/monitor.py wan_exp/cubic_text2video.py wan_exp/cubic_generate.py`
  - 结果：通过
  - 命令：`source ./.venv/bin/activate && python launch.py --model wan --method cubic`
  - 结果：通过，`results/wan/1.3B/cubic/two_anthropomorphic_cats_comfy/curvature_vis` 中仅保留 `curvature_overview.png`。

## 2026-04-08（straightness evaluation 验证）
- 目标模块：
  - `flexflow/straightness.py`
  - `wan_exp/cubic_text2video.py`
  - `wan_exp/cubic_generate.py`
- 变更摘要：
  - 新增仅用于 evaluation 的 `StraightnessEvaluator`，独立于加速主路径。
  - 使用“两锚点”方式计算：
    - endpoint prediction drift
    - finite-difference velocity curvature
    - drift proxy curvature
    - straightness score / ratio
  - 在 cubic 生成入口中默认启用 evaluation，并输出 `straightness_vis/straightness_overview.png`。
- 变更依据：
  - 用独立 evaluation 模块验证终点预测漂移与局部曲率的比例关系，而不是把它混入实际加速逻辑。
  - 继续保持与原始 WAN runtime 代码解耦。
- 验证命令与结果：
  - 命令：`python3 -m py_compile flexflow/monitor.py flexflow/straightness.py flexflow/__init__.py wan_exp/cubic_text2video.py wan_exp/cubic_generate.py`
  - 结果：通过
  - 命令：`source ./.venv/bin/activate && python launch.py --model wan --method cubic`
  - 结果：通过，生成 `results/wan/1.3B/cubic/two_anthropomorphic_cats_comfy/straightness_vis/straightness_overview.png`
  - 观测结果：anchor 处打印的 proxy-vs-FD 相关性约在 0.96~0.99，mean ratio 约在 0.68~1.06 区间，说明有强线性关系，但并非逐点完全等于 1。

## 2026-04-08（matplotlib 作图 + straightness 热力图总览）
- 目标模块：
  - `flexflow/monitor.py`
  - `flexflow/straightness.py`
- 变更摘要：
  - 将 `monitor` 的 overview 绘图从 PIL/imageio 改为 matplotlib。
  - 保持 `curvature_vis` 默认单图输出策略不变，仍仅导出 `curvature_overview.png`（可选重型文件开关保持默认关闭）。
  - 将 `straightness` 的作图完全改为 matplotlib，并新增可肉眼观察的热力图总览：`straightness_heatmap_overview.png`。
  - 保留 `straightness_overview.png` 作为指标关系图（散点+比值曲线+统计摘要）。
- 变更依据：
  - 满足“使用 matplotlib 作图”的要求。
  - 满足“straightness 需要热力图 overview”的可观测性要求。
- 验证命令与结果：
  - 命令：`python3 -m py_compile flexflow/monitor.py flexflow/straightness.py wan_exp/cubic_text2video.py wan_exp/cubic_generate.py`
  - 结果：通过
  - 命令：`source ./.venv/bin/activate && python launch.py --model wan --method cubic`
  - 结果：通过，输出文件包括：
    - `results/wan/1.3B/cubic/two_anthropomorphic_cats_comfy/curvature_vis/curvature_overview.png`
    - `results/wan/1.3B/cubic/two_anthropomorphic_cats_comfy/straightness_vis/straightness_overview.png`
    - `results/wan/1.3B/cubic/two_anthropomorphic_cats_comfy/straightness_vis/straightness_heatmap_overview.png`

## 2026-04-08（热力图强度按子图独立归一化）
- 目标模块：
  - `flexflow/monitor.py`
  - `flexflow/straightness.py`
- 变更摘要：
  - 将 `curvature_overview.png` 的热力图网格从全局 `vmin/vmax` 改为每个子图独立归一化。
  - 将 `straightness_heatmap_overview.png` 的热力图网格从全局 `vmin/vmax` 改为每个子图独立归一化。
- 变更依据：
  - 避免首张图动态范围过大时压缩后续子图对比度，提升肉眼可读性。
- 验证命令与结果：
  - 命令：`python3 -m py_compile flexflow/monitor.py flexflow/straightness.py`
  - 结果：通过

## 2026-04-09（groundtruth 曲率按定义(3.1)近似实现）
- 目标模块：
  - `flexflow/straightness.py`
  - `wan_exp/cubic_text2video.py`
- 变更摘要：
  - 将 straightness 评估中的 groundtruth 曲率从“简单有限差分代理”升级为按定义(3.1)近似计算。
  - 在 `StraightnessEvaluator.update(...)` 中新增 `velocity_fn` 回调输入，用于在评估侧重新计算速度场 $v(x,t)$。
  - 在 `_compute_groundtruth_curvature(...)` 中实现总导数近似：
    - 时间项：`partial_t = (v(x,t+eps_t) - v(x,t-eps_t)) / (2*eps_t)`
    - 对流项：沿速度方向的空间中心差分近似，并与速度幅值组合，构成 `(∇_x v)·v` 的近似
    - 最终曲率图：`||partial_t + advective||_2`
  - 新增基于 groundtruth 曲率图的可视化：`straightness_vis/groundtruth_curvature_heatmap_overview.png`。
  - `wan_exp/cubic_text2video.py` 中新增 `_unit_to_model_timestep` 与 `_make_velocity_fn`，在 anchor 评估时把 `velocity_fn` 传入 straightness evaluator。
  - straightness 日志字段由 `mean_fd` 改为 `mean_gt`，与 groundtruth 口径保持一致。
- 变更依据：
  - 对齐 [flexflow/docs/theory.md](../flexflow/docs/theory.md) 中式(3.1) $\frac{dv}{dt}=\partial_t v+(\nabla_x v)\cdot v$ 的定义，避免将 proxy 与“真值”使用同源近似导致验证偏乐观。
  - 遵循 [flexflow/docs/implement.md](../flexflow/docs/implement.md) 的边界：straightness 模块仅用于 evaluation，可独立于加速主路径演进。
- 受影响文件：
  - `flexflow/straightness.py`
  - `wan_exp/cubic_text2video.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`python3 -m py_compile flexflow/straightness.py wan_exp/cubic_text2video.py`
  - 结果：通过（无输出）
  - 命令：`source ./.venv/bin/activate && python launch.py --model wan --method cubic`
  - 结果：通过，端到端生成完成；新增输出 `results/wan/1.3B/cubic/two_anthropomorphic_cats_comfy/straightness_vis/groundtruth_curvature_heatmap_overview.png`。

## 2026-04-09（自适应四叉树分块实现）
- 目标模块：
  - `flexflow/types.py`（新建）
  - `flexflow/partition.py`（新建）
  - `flexflow/__init__.py`
  - `wan_exp/cubic_text2video.py`
- 变更摘要：
  - 新建 `flexflow/types.py`，定义核心数据结构：
    - `Phase`（枚举）：warmup / cruise_anchor / cruise_selective / cooldown
    - `Block`（数据类）：token 空间中的矩形块，具有 f/h/w 范围、token_count 属性、get_token_indices() 方法
    - `Partition`（数据类）：不重叠完全覆盖的分块方案，带 validate() 方法
    - `StepPlan`（数据类）：单步推理计划，包含 active/frozen blocks 信息
  - 新建 `flexflow/partition.py`，实现 `AdaptivePartitioner`：
    - `partition(Ft, Ht, Wt, curvature_map=None)`：主入口，若无曲率图返回均匀分块，否则执行自适应分裂
    - `_uniform_partition(Ft, Ht, Wt)`：生成均匀分块（block_size = max_block_size）
    - `_quadtree_split(...)`：递归四叉树分裂，分裂条件为 cv > cv_threshold 且块大小 > min_block_size
    - 支持 config.min_block_size, max_block_size, cv_threshold 参数
  - 更新 `flexflow/__init__.py`，导出新增的 Block, Partition, Phase, StepPlan, AdaptivePartitioner
  - 更新 `wan_exp/cubic_text2video.py`：
    - __init__ 中初始化 AdaptivePartitioner
    - 在 monitor.update() 成功返回 curvature_map 后立即调用 partitioner.partition()
    - 记录分块方案的 blocks 数量、token_grid 尺寸、逐块详细信息（block_id, 范围, token_count）
- 变更依据：
  - 遵循 [flexflow/docs/implement.md](../flexflow/docs/implement.md) 中 2.4（自适应四叉树分块）、4.2（types.py 规格）、4.3（partition.py 规格）
  - Block/Partition 等类型是模型无关的，放在 flexflow/ 作为可复用基础设施
  - Partitioner 的集成点在 anchor step 产出 curvature_map 后立即调用，无需修改原始 WAN runtime
- 受影响文件：
  - `flexflow/types.py`（新建，127 行）
  - `flexflow/partition.py`（新建，180 行）
  - `flexflow/__init__.py`（修改，+4 行）
  - `wan_exp/cubic_text2video.py`（修改，+40 行）
  - `worklog/cubic_worklog.md`（本条目）
- 验证命令与结果：
  - 命令：`python3 -m py_compile flexflow/types.py flexflow/partition.py flexflow/__init__.py wan_exp/cubic_text2video.py`
  - 结果：通过（无输出，语法正确）
  - 命令：`source ./.venv/bin/activate && python launch.py --model wan --method cubic`
  - 结果：**通过**（完整端到端生成）
    - 第7步(anchor)：92 blocks，token_grid=(21,30,52)，mean_curvature=1.385e+02
    - 第15步(anchor)：68 blocks，mean_curvature=1.855e+01
    - 第23步(anchor)：101 blocks，mean_curvature=9.838e+00
    - 第31步(anchor)：62 blocks，mean_curvature=4.991e+00
    - 第39步(anchor)：35 blocks，mean_curvature=2.551e+00
    - 可视化：curvature_vis, straightness_vis 已生成
    - 视频输出：./results/wan/1.3B/cubic/two_anthropomorphic_cats_comfy/cubic_two_anthropomorphic_cats_comfy_t2v-1.3B.mp4
  - **分块效果验证**：

## 2026-04-13（README 补充 CFG parallel 启用方式）
- 目标模块：`README.md`
- 变更摘要：
  - 在启动器章节新增“启用 CFG Parallel（--gpus 2）”小节。
  - 明确给出 WAN origin 与 WAN cubic 的 2 卡启动命令。
  - 补充 `--gpus 2` 与分布式环境变量、WAN 仅支持 2 卡 CFG 的关系说明。
- 变更依据：
  - 用户要求 README 明确包含 CFG parallel 的启用方式。
  - 对齐当前实现：`launch.py` 通过 `--gpus` 传入并由 WAN 侧以 `world_size == 2` 作为 CFG 并行约束。
- 受影响文件：
  - `README.md`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`source ./.venv/bin/activate && python launch.py --model wan --method cubic`
  - 结果：已在本次会话上下文中验证通过（Exit Code: 0）。

## 2026-04-13（WAN cubic 接入质量评估：新增 FID/FVD）
- 目标模块：
  - `utils/quality_metric.py`
  - `wan_exp/cubic_generate.py`
- 变更摘要：
  - 在质量评估器中新增了两类指标：
    - 图像/视频帧分布 `FID`（基于 InceptionV3 特征的 Fréchet 距离）
    - 视频 `FVD`（基于 R3D-18 视频特征的 Fréchet 距离）
  - 将视频评估流程扩展为输出 5 个指标：`psnr`、`ssim`、`lpips`、`fid`、`fvd`。
  - 为 `evaluate_image_quality(...)` 增加 `max_frames` 透传，确保视频评估帧数限制可用。
  - 在 `evaluate_quality_with_origin(...)` 中新增 `output_dir` 自动 reference 搜索逻辑：
    - 优先按 `.../<model_id>/cubic/<prompt_id>/` 对应到同级 `.../<model_id>/ori/<prompt_id>/`
    - 自动将生成文件前缀 `cubic_`/`{tag}_` 映射为 `ori_`
  - 在 WAN cubic 入口 `cubic_generate` 中完成工作流接入：
    - 生成视频并落盘后立即触发质量评估
    - 将 quality metrics 写入 `{tag}_params_metrics.json`（与 timing stats 同级）
- 变更依据：
  - 遵循 [flexflow/docs/implement.md](../flexflow/docs/implement.md) 中“解耦入口 `wan_exp/cubic_generate.py` 集成实验链路”的约束，不修改 origin WAN runtime。
  - 与文档的阶段化评估思想一致：在生成产物落盘后进行离线质量评测，避免干扰采样主循环。
- 受影响文件：
  - `utils/quality_metric.py`
  - `wan_exp/cubic_generate.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`source ./.venv/bin/activate && python launch.py --model wan --method cubic`
  - 结果：失败（依赖问题，非逻辑错误）
    - 报错：`ModuleNotFoundError: No module named 'skimage'`
    - 触发位置：`utils/quality_metric.py` 在模块导入阶段加载 `skimage.metrics`
  - 处理状态：按仓库约束未修改环境依赖（未安装/升级/降级包），已暂停等待用户决策。

## 2026-04-13（cubic_generate 超参数适配到运行链路）
- 目标模块：
  - `wan_exp/cubic_generate.py`
  - `wan_exp/cubic_text2video.py`
- 变更摘要：
  - 将 `cubic_generate.py` 顶部实验超参数接入 Cubic 运行配置：
    - `SPEEDUP -> cfg.target_speedup`
    - `WARMUP -> cfg.warmup_fraction`（按 `WARMUP / sample_steps` 转换）
    - `COOLDOWN -> cfg.cooldown_fraction`（按 `COOLDOWN / sample_steps` 转换）
    - `ANCHOR_INTERVAL -> cfg.anchor_interval`
  - 在 `cubic_generate.py` 增加覆盖后配置日志，便于确认本次实验使用的 optimizer/scheduler 参数。
  - 在 `CubicWanT2V` 中移除 phase 硬编码常量，改为优先读取 `config` 中的 `warmup_fraction/cooldown_fraction/anchor_interval`，确保入口参数真正影响 phase 判定与 anchor 调度。
- 变更依据：
  - 对齐 `flexflow/docs/implement.md` 2.1/2.2 中 phase 与 anchor 的配置化要求。
  - 对齐 `flexflow/docs/implement.md` 2.5 中 optimizer 目标速度参数（`target_speedup`）应可由实验入口控制。
- 受影响文件：
  - `wan_exp/cubic_generate.py`
  - `wan_exp/cubic_text2video.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`source ./.venv/bin/activate && python launch.py --model wan --method cubic`
  - 结果：通过（端到端生成完成，日志显示配置覆盖生效：`target_speedup=2.0 warmup_fraction=0.1 cooldown_fraction=0.1 anchor_interval=8`；结果文件与 `step_timing_summary.json` 正常输出）。

## 2026-04-10（selective phase 频率优化器 + 可视化统一）
- 目标模块：
  - `flexflow/optimizer.py`
  - `flexflow/visualizer.py`
  - `wan_exp/cubic_text2video.py`
- 变更摘要：
  - 新增 `UpdateFrequencyOptimizer`，在 anchor 处基于每个 partition 的 block 曲率分配更新间隔 `tau_b`。
  - 实现并接入公式：`tau_b ∝ kappa_b^{-alpha/(beta+1)}`，并按 `mean(1/tau_b)=1/target_speedup` 做归一化，再离散到 `[tau_min, tau_max]`。
  - 在 WAN T2V 的 anchor 流程中接入 optimizer，新增日志：`tau(min/max/mean)`、`mean_update_rate`、`achieved_speedup`。
  - 新建 `flexflow/visualizer.py` 作为统一可视化入口，收敛分块、曲率、straightness 导出逻辑。
  - 分块 frame strip 新增“更新频率可视化”：按 `tau` 给 block 边框着色并在框内标注 `tau`。
  - `flexflow/monitor.py` 与 `flexflow/straightness.py` 的导出方法改为调用统一 visualizer（保留兼容入口）。
  - `flexflow/partition_visualizer.py` 改为兼容重导出，避免旧导入路径失效。
- 变更依据：
  - 遵循 [flexflow/docs/implement.md](../flexflow/docs/implement.md) 的 2.5（`tau_b` 优化分配）与 2.2（anchor 步执行策略）。
  - 理论依据来自 [flexflow/docs/theory.md](../flexflow/docs/theory.md) 的局部曲率异质性与基于曲率控制跳步的误差-计算权衡思路。
  - 保持 WAN origin 代码不改，继续通过 `wan_exp/cubic_text2video.py` 解耦接入。
- 受影响文件：
  - `flexflow/optimizer.py`（新增）
  - `flexflow/visualizer.py`（新增）
  - `flexflow/__init__.py`
  - `flexflow/partition_visualizer.py`
  - `flexflow/monitor.py`
  - `flexflow/straightness.py`
  - `wan_exp/cubic_text2video.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`python3 -m py_compile flexflow/optimizer.py flexflow/visualizer.py flexflow/monitor.py flexflow/straightness.py flexflow/partition_visualizer.py flexflow/__init__.py wan_exp/cubic_text2video.py`
  - 结果：通过（无输出）
  - 命令：`source ./.venv/bin/activate && python launch.py --model wan --method cubic`
  - 结果：通过（完整端到端生成）
    - 新增 optimizer 日志示例：`[wan-optimizer] step=15 tau(min=1 max=4 mean=2.265) mean_update_rate=0.497475 achieved_speedup=2.010`
    - frame strip 继续输出到 `partition_vis/step_XXXX_frame_strip.png`，并包含 `tau` 叠加显示
    - 视频输出：`results/wan/1.3B/cubic/two_anthropomorphic_cats_comfy/cubic_two_anthropomorphic_cats_comfy_t2v-1.3B.mp4`

## 2026-04-10（删除旧 partition_visualizer 入口，统一 visualizer）
- 目标模块：
  - `flexflow/partition_visualizer.py`（删除）
  - `flexflow/visualizer.py`
  - `flexflow/__init__.py`
- 变更摘要：
  - 删除旧的 `partition_visualizer.py` 兼容入口，避免可视化逻辑双入口并存。
  - 保持 `PartitionVisualizer`、`CubicVisualizer`、`GradedPartitionVisualizer` 统一由 `flexflow/visualizer.py` 导出。
  - 确保包内 `__init__.py` 仅从 `visualizer.py` 进入可视化实现。
- 变更依据：
  - 按用户要求收敛可视化入口，避免后续 selective phase/调度器接线时出现旧路径混用。
  - 维持实现集中在一个文件，便于后续继续扩展 step plan / mask 可视化。
- 受影响文件：
  - `flexflow/partition_visualizer.py`（删除）
  - `flexflow/visualizer.py`
  - `flexflow/__init__.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`python3 -m py_compile flexflow/__init__.py flexflow/visualizer.py flexflow/optimizer.py flexflow/scheduler.py flexflow/monitor.py flexflow/straightness.py wan_exp/cubic_text2video.py`
  - 结果：通过（无语法错误输出）

## 2026-04-10（WAN selective sparse forward 接入 StepPlan）
- 目标模块：
  - `wan_exp/cubic_selective_forward.py`（新建）
  - `wan_exp/cubic_text2video.py`
- 变更摘要：
  - 新增 WAN 专用 `WanSelectiveForwardEngine`，将 `StepPlan` 真正用于 selective sparse forward。
  - 在 selective 步中实现 active/frozen token 分流：
    - active token 走完整 block 计算（self-attn + cross-attn + ffn）；
    - self-attn 内部使用 active Q + (active 更新后 + frozen 缓存) 的 K/V。
  - 实现了按 token 索引的 RoPE（indexed RoPE），用于 masked token 场景，避免沿用全序列连续 RoPE 假设。
  - KV cache 采用字典维护，key 采用 `(cfg_branch, layer_idx)`，分别维护 `k_cache` / `v_cache`；输出 cache 按 branch 维护。
  - selective 末端仅得到 active 输出后，用输出 cache 补全 frozen tokens，再执行 unpatchify。
  - 在 `CubicWanT2V` 中将 cond/uncond 两路 forward 改为统一走 selective 引擎（无 StepPlan 或 full step 自动走 full path 并刷新缓存）。
- 变更依据：
  - 对齐 [flexflow/docs/implement.md](../flexflow/docs/implement.md) 2.6（稀疏 attention）与 3.2（WAN 解耦入口）要求。
  - 遵循“不改 origin runtime 模块主体、通过新增模块解耦接入”的约束。
- 受影响文件：
  - `wan_exp/cubic_selective_forward.py`（新增）
  - `wan_exp/cubic_text2video.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`python3 -m py_compile wan_exp/cubic_selective_forward.py wan_exp/cubic_text2video.py flexflow/scheduler.py`
  - 结果：通过（无输出）
  - 命令：`source ./.venv/bin/activate && python launch.py --model wan --method cubic`
  - 结果：本轮两次触发均被会话侧“skip”未执行（非代码异常），未获得新的端到端运行日志。

## 2026-04-10（修复 selective 引擎 full 路径偏离 origin 的噪声问题）
- 目标模块：
  - `wan_exp/cubic_selective_forward.py`
- 问题定位：
  - 现象：首次 anchor 后 monitor/partition 可视化呈明显异常（近似噪声），即使在 full compute 场景也偏离历史正常分布。
  - 根因：在 selective 引擎中错误复用了全局 `e0.chunk(6)`，没有按每个 block 执行 `block.modulation + e_time` 后再 `chunk(6)`。
  - 影响：`WanAttentionBlock` 的调制项与 origin `block.forward` 不一致，导致 full 路径数值已偏移，进而污染 monitor 和后续分区。
- 变更摘要：
  - 将 `_selective_block_forward` 的调制逻辑改为与 origin 对齐：
    - `with autocast(float32): e0..e5 = (block.modulation + e_time).chunk(6, dim=1)`
  - 删除全局一次性 `e_chunks` 复用，改为每层 block 内实时计算。
- 变更依据：
  - 对齐 `wan_exp/wan/modules/model.py` 中 `WanAttentionBlock.forward` 的原始实现，保证 full 计算路径等价。
- 受影响文件：
  - `wan_exp/cubic_selective_forward.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`python3 -m py_compile wan_exp/cubic_selective_forward.py wan_exp/cubic_text2video.py`
  - 结果：通过
  - 命令：`source ./.venv/bin/activate && python launch.py --model wan --method cubic`
  - 结果：通过，且关键指标恢复到历史合理区间：
    - step=7：`curvature mean=1.385278e+02`，`chosen_block_size=(2,9,4)`，不再是异常 `(16,16,16)` 噪声分区
    - selective 步日志正常出现：`[wan-step-plan]` 与 `[wan-selective]`
    - 端到端输出完成：`cubic_two_anthropomorphic_cats_comfy_t2v-1.3B.mp4`

## 2026-04-13（适配 output_dir 新命名规则的 reference 自动查找）
- 目标模块：
  - `utils/quality_metric.py`
- 变更摘要：
  - 在 `infer_origin_path_from_output_dir(...)` 中增加了 prompt 目录候选逻辑。
  - 当 `output_dir` 使用新规则（例如 `.../cubic/<prompt_id>_2x` 或 `.../cubic/<prompt_id>_2.5x`）时，自动同时尝试：
    - 原目录名 `<prompt_id>_2x`
    - 去速率后缀目录名 `<prompt_id>`
  - 从而兼容 `cubic` 目录名带倍率后缀，而 `ori` 目录仍沿用原始 prompt_id 的场景。
- 变更依据：
  - 对齐当前实验入口 `wan_exp/cubic_generate.py` 的输出目录新命名策略（`{prompt_id}_{SPEEDUP}x`）。
  - 保持质量评测在“生成落盘后自动评估”链路中可稳定命中 baseline 视频。
- 受影响文件：
  - `utils/quality_metric.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`source ./.venv/bin/activate && python launch.py --model wan --method cubic`
  - 结果：待本轮命令执行后补充。

## 2026-04-10（优化模式关闭 straightness 默认计算）
- 目标模块：
  - `wan_exp/cubic_generate.py`
- 变更摘要：
  - 新增命令行开关 `--straightness_eval`，默认 `False`。
  - 将 `wan_t2v.generate(..., eval_straightness=...)` 从硬编码 `True` 改为读取该开关。
  - 这样优化模式下默认仅使用轻量 curvature 代理，不再默认执行 straightness ground-truth 评估链路。
- 变更依据：
  - straightness 属于验证/对照（ground-truth）链路，不应在性能优化模式默认开启，避免额外前向计算开销。
- 受影响文件：
  - `wan_exp/cubic_generate.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`python3 -m py_compile wan_exp/cubic_generate.py wan_exp/cubic_text2video.py`
  - 结果：通过（无语法错误输出）
    - 动态调整：块数从 92 → 68 → 101 → 62 → 35，随曲率变化（138.5 → 18.5 → 9.8 → 5.0 → 2.6）
    - 未见崩溃或数据不一致，partition.validate() 通过
    - log 显示所有 anchor blocks 计数、范围信息正常输出


## 2026-04-09（分区口径重构：全局 blocksize + 均匀 3D 切分）
- 目标模块：
  - `flexflow/partition.py`
  - `flexflow/partition_visualizer.py`
  - `flexflow/types.py`
  - `wan_exp/cubic_text2video.py`
  - `flexflow/docs/implement.md`
- 变更摘要：
  - 将分区策略从“递归细分/分级思路”收敛为单一口径：
    - 仅决定一个全局 `blocksize(f,h,w)`；
    - 按该 blocksize 在 `(Ft,Ht,Wt)` 上均匀切分（边界截断）；
    - 选择依据是“均匀切分后各 block 内曲率尽量接近”（使用 block 内 `cv` 统计）。
  - 在 `partition.py` 中新增候选 blocksize 搜索与一致性打分：
    - 候选打分包含 `mean_block_cv` 与 `p90_block_cv`；
    - 在一致性约束下优先更大 block，避免不必要的小块爆炸。
  - `Partition` 增加 `selected_block_size` 字段，便于日志与可视化链路复核。
  - `partition_visualizer.py` 调整为每个 anchor 输出：
    - `step_XXXX_frame_strip.png`（block 边界叠加）
    - `step_XXXX_partition_summary.txt`（blocksize 与块内统计）
  - 可视化目录不再清理历史文件，保留所有 anchor 的 partition overview。
  - `wan_exp/cubic_text2video.py` 日志改为重点打印：
    - `chosen_block_size`
    - `blocks` / `token_stats`
    - `block_cv(mean/p90)`
- 变更依据：
  - 对齐用户口径：不做评级，不做按语义区域聚类；只做“全局 blocksize 决策 + 均匀切分”。
  - 对齐 `jano_blockmanager` 的规则网格思路：先确定 block size，再全局规则切块。
  - 保证每个 anchor 都有可追踪的 overview 产物，便于横向对比。
- 受影响文件：
  - `flexflow/partition.py`
  - `flexflow/partition_visualizer.py`
  - `flexflow/types.py`
  - `wan_exp/cubic_text2video.py`
  - `flexflow/docs/implement.md`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`python3 -m py_compile flexflow/types.py flexflow/partition.py flexflow/partition_visualizer.py flexflow/__init__.py wan_exp/cubic_text2video.py`
  - 结果：通过（无输出）
  - 命令：`source ./.venv/bin/activate && python launch.py --model wan --method cubic`
  - 结果：通过（见 `uniform_blocksize_test.log` / `partition_overview_test.log`）
  - 关键日志样例：
    - step=7: `chosen_block_size=(2, 9, 4)`, `blocks=572`, `block_cv(mean=0.4427 p90=0.7991)`
    - step=15: `chosen_block_size=(2, 4, 9)`, `blocks=528`, `block_cv(mean=0.4713 p90=0.7759)`
    - step=23: `chosen_block_size=(4, 4, 8)`, `blocks=336`, `block_cv(mean=0.5334 p90=0.8730)`
    - step=39: `chosen_block_size=(2, 4, 9)`, `blocks=528`, `block_cv(mean=0.4287 p90=0.6542)`
  - 可视化产物：
    - `results/wan/1.3B/cubic/two_anthropomorphic_cats_comfy/partition_vis/step_XXXX_frame_strip.png`
    - `results/wan/1.3B/cubic/two_anthropomorphic_cats_comfy/partition_vis/step_XXXX_partition_summary.txt`

## 2026-04-13（修复 CFG 下两分支 cubic 决策不一致）
- 目标模块：
  - `wan_exp/cubic_text2video.py`
- 变更摘要：
  - 修复 2 卡 CFG 路径中仅 `rank0` 计算 partition/frequency/stepplan 的问题。
  - 现在在 anchor/warmup-tail 上，两卡都会执行：
    - `partitioner.partition(...)`
    - `frequency_optimizer.optimize(...)`
    - `step_scheduler.build_step_plan(...)`
  - 仅把日志与可视化保留在 `rank0`，避免重复输出。
  - CFG 路径判断改为 `get_cp_worldsize()==2` 后再获取 `cp_group`，避免单卡路径直接访问 CFG group。
- 变更依据：
  - CFG 并行下 cond/uncond 两分支必须共享同一 step 级 cubic 决策口径，否则会出现一支 selective、另一支 full 的行为漂移，造成质量下降与吞吐受限。
- 受影响文件：
  - `wan_exp/cubic_text2video.py`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`source ./.venv/bin/activate && python launch.py --model wan --method cubic --gpus 2`
  - 结果：通过（端到端生成完成，输出视频与指标文件正常）。
  - 关键观测：`step_timing_summary.json` 中 `avg_total_ms` 约从 3248ms 降至 1840ms，`avg_forward_ms` 约从 3080ms 降至 1673ms。

## 2026-04-13（README 补充输出文件位置与含义）
- 目标模块：`README.md`
- 变更摘要：
  - 新增“输出文件位置与含义”章节，覆盖 WAN cubic / WAN ori / FireRed ori / Flux2-klein ori 的默认输出目录。
  - 逐项说明了视频、参数与指标 JSON、计时统计、可视化产物（curvature/straightness）的文件名和用途。
  - 增补质量指标说明（PSNR/SSIM/LPIPS/FID/FVD）与使用场景。
  - 保持并强调 CFG parallel 启用方式 `--gpus 2`。
- 变更依据：
  - 用户要求 README 补齐“所有输出文件的位置和含义”，并包含可视化/时间/质量/视频本身。
  - 对齐当前 WAN cubic 实际产物与现有结果目录结构。
- 受影响文件：
  - `README.md`
  - `worklog/cubic_worklog.md`
- 验证命令与结果：
  - 命令：`source ./.venv/bin/activate && python launch.py --model wan --method cubic`
  - 结果：会话上下文已有最近一次成功运行记录（Exit Code: 0），本次为文档改动，未引入运行逻辑变更。

## 2026-04-15（flux2k selective attention：双 processor 的 RoPE 后 KV cache 更新与补全）
- 日期：2026-04-15
- 目标模块：`flux2k_exp/flux2k_cubic/attn.py`
- 变更摘要：
  - 在 `attn.py` 中新增两类 selective attention processor：
    - `Flux2CubicKVCacheAttnProcessor`：用于 `Flux2AttnProcessor` 对应的双流注意力，保持 `encoder_hidden_states` 全量，仅对 image token 的 K/V 进行缓存更新与补全。
    - `Flux2CubicKVCacheParallelSelfAttnProcessor`：用于 `Flux2ParallelSelfAttnProcessor` 对应的单流并行注意力，在无显式 encoder 输入的场景下，对整段序列执行 frozen K/V 补全。
  - 严格按要求把 KV 更新/拼接放在 `apply_rotary_emb(...)` 之后，且尽量不改动原 attention 主流程（QKV 投影、norm、dispatch、输出投影保持一致）。
  - 增加 `_call_cache_update(...)` 作为最小适配层，兼容 `_update_and_get_full_kv_cache` / `_update_and_get_kv_cache` / `_update_and_get_image_kv_cache` 不同缓存接口命名。
- 变更依据：
  - 对齐 `flexflow/docs/implement.md` 2.6：cruise selective 阶段应通过 per-layer KV cache 让 frozen token 参与 K/V 端，而 active token 正常计算。
  - 对齐 `flexflow/docs/theory.md` §5 的 ZOH 复用思想：冻结区域通过缓存维持一致上下文，减少重复计算。
- 触达文件：
  - `flux2k_exp/flux2k_cubic/attn.py`
  - `worklog/cubic_worklog.md`
- 验证命令：
  - `source ./.venv/bin/activate`
  - `python launch.py --model flux2k --method cubic`
- 验证结果：
  - 命令执行通过（Exit Code 0）。
  - 流程完成 warmup + 3 seed 生成，计时统计正常输出（`load_pipeline`/`inference`/`step_xxxx_total`/`save_image`）。

## 2026-04-15（flux2k selective+kv 回退修复：按层区分 KV cache 长度校验）
- 日期：2026-04-15
- 目标模块：`flux2k_exp/flux2k_cubic/cubic_selective_forward.py`
- 变更摘要：
  - 修复 `use_kv_cache=True` 时 selective 步被误判为“KV 缓存不完整”并回退 full forward 的问题。
  - 将 `_has_kv_cache_for_all_layers(...)` 的长度校验从单一 `total_tokens` 改为按层类型分别校验：
    - `joint` 层期望长度为 `image_tokens`；
    - `single` 层期望长度为 `encoder_tokens + image_tokens`。
  - 在 selective 分支调用校验函数时显式传入 `image_tokens` 与 `encoder_tokens`，确保与 processor 实际缓存布局一致。
- 变更依据：
  - 对齐 `flexflow/docs/implement.md` 2.6：selective+KV 路径中 single/joint 层可拥有不同的有效序列长度语义，不应使用单一长度阈值统一判定。
- 触达文件：
  - `flux2k_exp/flux2k_cubic/cubic_selective_forward.py`
  - `worklog/cubic_worklog.md`
- 验证命令：
  - `source ./.venv/bin/activate`
  - `python launch.py --model flux2k --method cubic -- --use_kv_cache`
- 验证结果：
  - 运行通过（Exit Code 0）。
  - selective+kv 生效：step2/step3 仍为 `full_compute=False`；且日志显示 `query` 为活跃子序列（如 2048/3328），`key/value` 为补全后全序列（4608），符合“selective forward + frozen KV cache”的预期。
  - 输出目录后缀生效：结果写入 `results/flux2/flux2-klein/cubic_kv/...`。
