# Cubic 工作日志

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
