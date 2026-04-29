"""
FireRed + Jano 推理入口

Jano 针对 FireRed 的适配要点：
  - FireRed 序列 = noise_latent_tokens + ref_image_latent_tokens
  - 噪声 latent tokens：warmup 阶段收集 → DynamicAnalyzer 时空动态性分析
    → 三级掩码 (static / medium / active)
  - 参考图 latent tokens：一次 FFT 复杂度分析 → 二级掩码 (static / active)
  - step_level 0/3: 全量计算；2: medium+active；1: active

用法（与 PAB/ToCa 一致）：
    cd firered_exp
    python jano_generate.py [args...]

与 ori_generate.py 对比质量：
    将 --disable_quality_eval 去掉，脚本会自动寻找同目录下 ori 基准图计算 PSNR/SSIM/LPIPS。
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
from pathlib import Path

# Ensure the firered_exp directory (where the `jano` package lives) is on sys.path.
_firered_exp_dir = os.path.dirname(os.path.abspath(__file__))
if _firered_exp_dir not in sys.path:
    sys.path.insert(0, _firered_exp_dir)

import torch
from PIL import Image

# Jano monkey patch（导入即生效）
import baseline.jano_firered  # noqa: F401

from firered.pipeline import QwenImageEditPlusPipeline
from firered.transformer import QwenImageTransformer2DModel

from utils.launch_defaults import resolve_model_path
from utils.quality_metric import evaluate_quality_with_origin
from utils.results import get_prompt_id, save_params_and_metrics
from utils.timer import get_time_statistics_dict, get_timer, init_timer, print_time_statistics

# ── 默认配置 ──────────────────────────────────────────────────────────────────
IMAGE_PATH = "/home/fit/zhaijdcyy/WORK/Cubic/firered_exp/examples/cola.png"
PROMPT = (
    "Transform the object into a realistic miniature product by carefully holding it "
    "between your thumb and forefinger."
)
TAG = "jano"

# Jano 超参（与 Jano WAN 默认配置对齐）
JANO_WARMUP_STEPS        = 5     # warmup 阶段步数（收集噪声 latent）
JANO_COOLDOWN_STEPS      = 5     # cooldown 末尾全量步数
JANO_STATIC_INTERVAL     = 6     # 每隔几步做一次全量 + 缓存更新（level-3）
JANO_MEDIUM_INTERVAL     = 3     # 每隔几步做一次 medium 计算（level-2）
JANO_STATIC_THRESH       = 0.2   # 噪声 token 判定为 static 的动态性阈值
JANO_MEDIUM_THRESH       = 0.6   # 噪声 token 判定为 medium 的动态性阈值
JANO_REF_THRESH          = 0.5   # 参考图 token 判定为 active 的 FFT 能量阈值
JANO_ANALYZE_BLOCK_SIZE  = (1, 4, 4)   # 分析 block 尺寸 (T, H, W)，T 必须为 1（图像）

quality_eval = True


# ── 日志 ──────────────────────────────────────────────────────────────────────

def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
        force=True,
    )


# ── 参数解析 ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FireRed Jano inference script")
    parser.add_argument("--model_path",  type=str,  default=None)
    parser.add_argument("--input_image", type=Path, nargs="+", default=[Path(IMAGE_PATH)])
    parser.add_argument("--output_image", type=Path, default=None)
    parser.add_argument("--output_dir",   type=Path, default=None)
    parser.add_argument("--prompt",       type=str,  default=PROMPT)
    parser.add_argument("--seed",         type=int,  default=49)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)

    # Jano 超参
    parser.add_argument("--jano_warmup_steps",       type=int,   default=JANO_WARMUP_STEPS)
    parser.add_argument("--jano_cooldown_steps",     type=int,   default=JANO_COOLDOWN_STEPS)
    parser.add_argument("--jano_static_interval",    type=int,   default=JANO_STATIC_INTERVAL,
                        help="每隔几步做一次全量+缓存更新（level-3），0 表示关闭稀疏计算")
    parser.add_argument("--jano_medium_interval",    type=int,   default=JANO_MEDIUM_INTERVAL,
                        help="每隔几步做一次 medium 计算（level-2），0 表示关闭")
    parser.add_argument("--jano_static_thresh",      type=float, default=JANO_STATIC_THRESH,
                        help="噪声 token 动态性 < static_thresh 视为 static")
    parser.add_argument("--jano_medium_thresh",      type=float, default=JANO_MEDIUM_THRESH,
                        help="噪声 token 动态性 ∈ [static_thresh, medium_thresh] 视为 medium")
    parser.add_argument("--jano_ref_thresh",         type=float, default=JANO_REF_THRESH,
                        help="参考图 token FFT 能量 > ref_thresh 视为 active")
    parser.add_argument("--jano_analyze_block_h",    type=int,   default=JANO_ANALYZE_BLOCK_SIZE[1],
                        help="分析 block 高度（latent pixel 单位）")
    parser.add_argument("--jano_analyze_block_w",    type=int,   default=JANO_ANALYZE_BLOCK_SIZE[2],
                        help="分析 block 宽度（latent pixel 单位）")

    parser.add_argument(
        "--disable_quality_eval", action="store_true", default=not quality_eval,
        help="关闭相对 ori 基准的质量评测"
    )
    parser.add_argument(
        "--disable_progress_bar", action="store_true", default=False
    )
    return parser.parse_args()


# ── Pipeline 加载 ─────────────────────────────────────────────────────────────

def load_pipeline(model_path: str, disable_progress_bar: bool) -> QwenImageEditPlusPipeline:
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        transformer=QwenImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        ),
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True if disable_progress_bar else None)
    return pipe


# ── 输出路径解析 ──────────────────────────────────────────────────────────────

def _resolve_output_paths(
    model_path: str,
    prompt: str,
    static_interval: int,
    medium_interval: int,
    output_dir: Path | None,
    output_image: Path | None,
) -> tuple[Path, Path, str]:
    model_id   = Path(model_path).name
    prompt_id  = get_prompt_id(prompt)
    speed_tag  = f"si{static_interval}_mi{medium_interval}"
    resolved_dir = (
        output_dir
        or Path("results") / "firered" / model_id / TAG / f"{prompt_id}_{speed_tag}"
    )
    resolved_name  = f"{TAG}_{prompt_id}_{model_id}.png"
    resolved_image = output_image or (resolved_dir / resolved_name)
    return resolved_dir, resolved_image, model_id


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_logging()
    init_timer()
    args = parse_args()
    args.model_path = resolve_model_path(args.model_path, "--model_path")

    # 开始 generate_e2e 计时（覆盖整个生成流程）
    timer_context = get_timer("generate_e2e")
    timer_context.__enter__()

    analyze_block_size = (1, args.jano_analyze_block_h, args.jano_analyze_block_w)

    output_dir, output_image_path, model_id = _resolve_output_paths(
        args.model_path, args.prompt,
        args.jano_static_interval, args.jano_medium_interval,
        args.output_dir, args.output_image,
    )
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 打印配置摘要 ──────────────────────────────────────────────
    sep = "=" * 60
    logging.info(sep)
    logging.info("  FireRed + Jano Generation")
    logging.info(sep)
    logging.info("  Model          : %s", args.model_path)
    logging.info("  Prompt         : %s", args.prompt)
    logging.info("  Steps          : %d  Seed: %d  CFG: %.1f",
                 args.num_inference_steps, args.seed, args.true_cfg_scale)
    logging.info("  Jano config:")
    logging.info("    warmup_steps    = %d", args.jano_warmup_steps)
    logging.info("    cooldown_steps  = %d", args.jano_cooldown_steps)
    logging.info("    static_interval = %d  (level-3 full+update)",  args.jano_static_interval)
    logging.info("    medium_interval = %d  (level-2 medium)",       args.jano_medium_interval)
    logging.info("    static_thresh   = %.2f", args.jano_static_thresh)
    logging.info("    medium_thresh   = %.2f", args.jano_medium_thresh)
    logging.info("    ref_thresh      = %.2f", args.jano_ref_thresh)
    logging.info("    analyze_block   = %s", str(analyze_block_size))
    logging.info("  Output         : %s", output_image_path)
    logging.info(sep)

    # ── 加载 Pipeline ─────────────────────────────────────────────
    print("Loading pipeline...", flush=True)
    with get_timer("load_pipeline"):
        pipeline = load_pipeline(args.model_path, args.disable_progress_bar)
    print("Pipeline loaded.", flush=True)

    # ── 配置 Jano 超参到 transformer ─────────────────────────────
    tr = pipeline.transformer
    tr.enable_jano              = True
    tr.num_steps                = args.num_inference_steps
    tr.jano_warmup_steps        = args.jano_warmup_steps
    tr.jano_cooldown_steps      = args.jano_cooldown_steps
    tr.jano_static_interval     = args.jano_static_interval
    tr.jano_medium_interval     = args.jano_medium_interval
    tr.jano_static_thresh       = args.jano_static_thresh
    tr.jano_medium_thresh       = args.jano_medium_thresh
    tr.jano_ref_thresh          = args.jano_ref_thresh
    tr.jano_analyze_block_size  = analyze_block_size

    if hasattr(tr, "reset_jano_state"):
        tr.reset_jano_state()

    logging.info(
        "[firered-jano] prompt_id=%s  steps=%d  warmup=%d  cooldown=%d  "
        "si=%d  mi=%d  st=%.2f  mt=%.2f  rt=%.2f  block=%s  output=%s",
        get_prompt_id(args.prompt),
        args.num_inference_steps,
        args.jano_warmup_steps,
        args.jano_cooldown_steps,
        args.jano_static_interval,
        args.jano_medium_interval,
        args.jano_static_thresh,
        args.jano_medium_thresh,
        args.jano_ref_thresh,
        str(analyze_block_size),
        output_dir,
    )

    # ── 加载输入图像 ─────────────────────────────────────────────
    with get_timer("load_images"):
        images = [Image.open(p).convert("RGB") for p in args.input_image]
    logging.info("[firered-jano] loaded_images=%d", len(images))

    inputs = {
        "image":               images,
        "prompt":              args.prompt,
        "generator":           torch.Generator(device="cuda").manual_seed(args.seed),
        "true_cfg_scale":      args.true_cfg_scale,
        "negative_prompt":     " ",
        "num_inference_steps": args.num_inference_steps,
        "num_images_per_prompt": 1,
    }

    # ── 推理 ─────────────────────────────────────────────────────
    with get_timer("inference_first"):
        with torch.inference_mode():
            result = pipeline(**inputs)

    output_image = result.images[0]
    with get_timer("save_image"):
        output_image.save(output_image_path)

    # ── 质量评测 ─────────────────────────────────────────────────
    quality_metrics = None
    if not args.disable_quality_eval:
        with get_timer("quality_eval"):
            quality_result = evaluate_quality_with_origin(
                image_path=str(output_image_path),
                tag=TAG,
                save_metrics=False,
                baseline_path=None,
                output_dir=str(output_dir),
            )
        quality_metrics = quality_result["metrics"] if quality_result is not None else None

    print("Image saved at:", output_image_path.resolve(), flush=True)
    print_time_statistics()

    # ── 参数与指标保存 ────────────────────────────────────────────
    params = {
        "model":               "firered",
        "method":              TAG,
        "model_id":            model_id,
        "model_path":          args.model_path,
        "prompt":              args.prompt,
        "input_image":         [str(p) for p in args.input_image],
        "output_image":        str(output_image_path),
        "seed":                args.seed,
        "true_cfg_scale":      args.true_cfg_scale,
        "num_inference_steps": args.num_inference_steps,
        "jano_warmup_steps":   args.jano_warmup_steps,
        "jano_cooldown_steps": args.jano_cooldown_steps,
        "jano_static_interval": args.jano_static_interval,
        "jano_medium_interval": args.jano_medium_interval,
        "jano_static_thresh":   args.jano_static_thresh,
        "jano_medium_thresh":   args.jano_medium_thresh,
        "jano_ref_thresh":      args.jano_ref_thresh,
        "jano_analyze_block":   list(analyze_block_size),
        "quality_eval":         not args.disable_quality_eval,
    }
    # 结束 generate_e2e 计时
    timer_context.__exit__(None, None, None)

    params_path = save_params_and_metrics(
        str(output_dir), TAG, params,
        get_time_statistics_dict(),
        quality_metrics=quality_metrics,
    )
    print(f"Params & metrics saved to {params_path}", flush=True)


if __name__ == "__main__":
    main()
