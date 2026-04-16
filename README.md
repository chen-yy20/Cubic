# cubic

本仓库用于统一管理多模型推理实验（origin / cubic），当前支持通过 `launch.py` 启动 WAN、Flux1、FireRed、Flux2-klein 等实验脚本。

## 1. 环境配置

### 1.1 基础要求

- Linux + NVIDIA GPU
- Python 3.12
- 建议使用 `uv` 管理依赖

### 1.2 创建虚拟环境

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
```

### 1.3 安装依赖

优先直接使用仓库锁定依赖：

```bash
uv sync
```

若你需要手动安装，至少保证 PyTorch + CUDA 版本匹配：

```bash
uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/cu131
uv add diffusers transformers accelerate safetensors huggingface_hub datasets peft scipy einops opencv-python
```

### 1.4 快速验证

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -c "import diffusers; print(diffusers.__version__)"
```

## 2. 统一启动器 launch.py

`launch.py` 的作用是：根据 `--model` + `--method` 自动映射到对应脚本，并透传后续参数。

通用格式：

```bash
python launch.py --model <model> --method <method> [--launcher python|srun] [--gpus N] -- <script args>
```

说明：

- `--launcher` 默认是 `srun`，本地单机调试建议显式加 `--launcher python`。
- `--gpus` 仅在 `srun` 模式生效。
- `--` 之后的参数会原样传给目标脚本。

当前内置映射：

- `wan + ori` -> `wan_exp/ori_generate.py`
- `wan + cubic` -> `wan_exp/cubic_generate.py`
- `flux + ori` -> `flux1_exp/ori_generate.py`
- `flux + cubic` -> `flux1_exp/cubic_generate.py`
- `firered + ori` -> `firered_exp/ori_generate.py`
- `flux2k + ori` -> `flux2k_exp/ori_generate.py`

### 2.1 启用 CFG Parallel（`--gpus 2`）

WAN 的 origin/cubic 都支持 2 卡 CFG 并行。通过 `launch.py` 启用方式如下：

```bash
python launch.py --model wan --method ori --gpus 2
python launch.py --model wan --method cubic --gpus 2
```

说明：

- 该参数通过 `pysrun.sh` 传递为分布式环境变量（`WORLD_SIZE=2`）。
- WAN 侧当前只支持 2 卡 CFG 组（脚本中有 `world_size == 2` 断言）。
- 若只想单卡本地调试，请使用 `--launcher python`。

## 3. 如何执行各模型 origin

先激活环境：

```bash
source .venv/bin/activate
```

### 3.1 WAN origin

```bash
python launch.py --model wan --method ori --launcher python
```

2 卡（默认走 `srun`）示例：

```bash
python launch.py --model wan --method ori --gpus 2
```

### 3.2 FireRed origin

```bash
python launch.py --model firered --method ori --launcher python
```

覆盖 prompt / 推理步数示例：

```bash
python launch.py --model firered --method ori --launcher python -- --prompt "Transform to a toy style" --num_inference_steps 30
```

### 3.3 Flux2-klein origin

```bash
python launch.py --model flux2k --method ori --launcher python
```

覆盖分辨率示例：

```bash
python launch.py --model flux2k --method ori --launcher python -- --height 768 --width 768
```

### 3.4 Flux1 origin

```bash
python launch.py --model flux --method ori --launcher python
```

## 4. Cubic 实验

### 4.1 WAN cubic

基础命令：

```bash
source .venv/bin/activate
python launch.py --model wan --method cubic --launcher python
```

2 卡 CFG 并行示例：

```bash
source .venv/bin/activate
python launch.py --model wan --method cubic --gpus 2
```

### 4.2 Flux1 cubic

基础命令：

```bash
source .venv/bin/activate
python launch.py --model flux --method cubic
```

当前 Flux1 cubic 已接入：

- 三阶段推理（warmup / cruise-anchor / cruise-selective / cooldown）
- anchor 步监测、分块与频率分配
- selective forward 路径（phase3）
- 曲率与分块可视化导出（可开关）
- 生成后质量评测（按 seed 评测并汇总 mean）

## 5. 输出文件位置与含义

### 5.1 WAN cubic 输出目录

默认输出根目录：

```text
results/wan/<model_id>/cubic/<prompt_id>_<speedup>x/
```

常见文件与含义：

- `cubic_<prompt_id>_<task>.mp4`
	- 含义：最终生成的视频文件（核心产物）。
- `cubic_params_metrics.json`
	- 含义：本次实验参数快照 + 质量指标汇总 + 计时统计汇总。
	- 作用：做实验对比时的主索引文件。
- `timing_stats.txt`
	- 含义：计时器统计明细（如 `generate_e2e`、`selective_forward`、`anchor_step_total`）。
	- 作用：分析加速收益和瓶颈。
- `curvature_vis/curvature_overview.png`（当开启/保留可视化导出时）
	- 含义：anchor 曲率可视化总览图。
	- 作用：观察不同 step 的曲率分布变化。
- `straightness_vis/straightness_overview.png`（开启 `--straightness_eval true` 时）
	- 含义：straightness 指标关系图（统计散点/曲线总览）。
- `straightness_vis/straightness_heatmap_overview.png`（开启 `--straightness_eval true` 时）
	- 含义：straightness 热力图总览。
- `straightness_vis/groundtruth_curvature_heatmap_overview.png`（开启 `--straightness_eval true` 时）
	- 含义：groundtruth 曲率热力图总览，用于与 proxy/估计值对照。

### 5.2 WAN origin 输出目录

```text
results/wan/<model_id>/ori/<prompt_id>/
```

常见文件与含义：

- `ori_<prompt_id>_<task>.mp4`：origin 基线视频。
- `ori_params_metrics.json`：origin 的参数、计时与质量指标汇总。

### 5.3 FireRed origin 输出目录

```text
results/firered/<model_id>/ori/<prompt_id>/
```

常见文件与含义：

- `ori_<prompt_id>_<model_id>.png`：编辑后图片。
- `ori_params_metrics.json`：参数与计时统计（FireRed 默认不做视频质量评估）。

### 5.4 Flux2-klein origin 输出目录

```text
results/flux2/<model_id>/ori/<prompt_id>/
```

常见文件与含义：

- `ori_<prompt_id>_<model_id>.png`：生成图片。
- `ori_params_metrics.json`：参数与计时统计。

### 5.5 Flux1 cubic 输出目录

```text
results/flux/cubic/<prompt_id>_<speedup>x/
```

常见文件与含义：

- `cubic_<prompt_id>_seed<seed>.png`
	- 含义：每个 seed 的最终生成图片。
- `cubic_params_metrics.json`
	- 含义：参数、计时统计、质量评测结果（`per_seed` 与 `mean`）。
- `curvature_vis/curvature_overview_seed<seed>.png`（开启可视化时）
	- 含义：按 seed 导出的曲率总览图。
- `partition_vis/step_XXXX_frame_strip_seed<seed>.png`（开启可视化时）
	- 含义：anchor 步分块与 tau 可视化条带图。
- `partition_vis/step_XXXX_partition_summary_seed<seed>.txt`（开启可视化时）
	- 含义：对应 anchor 步的分块统计与频率分配摘要。

### 5.6 质量指标（WAN / Flux）说明

WAN 与 Flux 路径都可调用质量评估并写入日志/JSON，常见指标：

- `PSNR` / `SSIM`：保真度指标。
- `LPIPS`：感知差异指标（越低通常越接近 baseline）。
- `FID` / `FVD`：分布级图像/视频质量距离指标。
	- 图像任务（Flux）通常仅有 `FID`，`FVD` 可能为 `null`。

## 6. 重点超参数说明（WAN cubic）

WAN cubic 的实验入口是 `wan_exp/cubic_generate.py`，核心超参数在脚本顶部定义。

### 5.1 `WARMUP`

- 含义：前期全量计算步数，用于建立初始曲率估计。
- 对应实现：`warmup_fraction = WARMUP / sample_steps`。
- 理解：越大越稳，但可加速区间变短。

### 5.2 `COOLDOWN`

- 含义：末期全量计算步数，保证收敛与最终质量。
- 对应实现：`cooldown_fraction = COOLDOWN / sample_steps`。
- 理解：越大通常质量越稳，速度收益会下降。

### 5.3 `SPEEDUP`

- 含义：目标加速比（target speedup）。
- 对应实现：`cfg.target_speedup = SPEEDUP`，用于约束 cruise 阶段平均更新率。
- 理解：数值越大，理论上越激进；过大可能导致质量下降。

### 5.4 `enable_visualization`

- CLI 参数：`--enable_visualization true|false`。
- 含义：是否导出 cubic 分析可视化（例如 phase / token 激活相关诊断信息）。
- 建议：调参与排障时开启，批量跑基准时关闭以减少额外开销。

示例：

```bash
python launch.py --model wan --method cubic --launcher python -- --enable_visualization true
```

### 5.5 `straightness_eval`

- CLI 参数：`--straightness_eval true|false`。
- 含义：是否启用 straightness 评估链路（额外开销较大）。
- 建议：仅在验证阶段开启；日常 speed benchmark 关闭。

示例：

```bash
python launch.py --model wan --method cubic --launcher python -- --straightness_eval true
```

## 7. 参数与理论的对应关系

根据 `cubic_core/docs/implement.md`：

- 三阶段划分：Warmup / Cruise / Cooldown。
- 其中 $K_w = \lfloor K \cdot warmup\_fraction \rfloor$，$K_c = \lfloor K \cdot cooldown\_fraction \rfloor$。
- `target_speedup` 约束 cruise 阶段的平均更新率（对应块级更新频率分配）。

根据 `cubic_core/docs/theory.md`：

- straightness 用于刻画局部轨迹弯曲程度，指导哪些区域可跳步复用、哪些区域需要更频繁更新。
