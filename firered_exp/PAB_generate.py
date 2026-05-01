"""FireRed + PAB 推理入口：与 teacache/cubic/ori 一致的计时与相对 ori 的质量评估。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image

import baseline.PAB_firered  # noqa: F401 — monkey patch FireRed Transformer
from firered.pipeline import QwenImageEditPlusPipeline
from firered.transformer import QwenImageTransformer2DModel

from utils.launch_defaults import resolve_model_path
from utils.quality_metric import evaluate_quality_with_origin
from utils.results import get_prompt_id, save_params_and_metrics
from utils.timer import get_time_statistics_dict, init_timer, print_time_statistics, save_time_stats_to_file

IMAGE_PATH = "/home/fit/zhaijdcyy/WORK/Cubic/firered_exp/examples/cola.png"
IMAGE_PATH = "/home/chenyy/WORK/cyy/cubic/firered_exp/examples/cola.png"

PROMPT = (
    "Transform the object into a realistic miniature product by carefully holding it "
    "between your thumb and forefinger."
)
TAG = "pab"
REQUIRED_TIMER_NAMES = [
    "generate_e2e",
    "dit_forward",
]

# PAB 默认超参
PAB_WARMUP_STEPS = 5
PAB_COOLDOWN_STEPS_OFFSET = 5   # 从末尾往前保留几步做全量计算
PAB_SKIP_RANGE = 4               # attention 复用间隔（每隔几步重算一次）

quality_eval = True


def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FireRed PAB inference script")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--input_image", type=Path, nargs="+", default=[Path(IMAGE_PATH)])
    parser.add_argument("--output_image", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--seed", type=int, default=49, help="与 ori/teacache/cubic 默认一致，便于质量对比")
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument(
        "--pab_warmup_steps",
        type=int,
        default=PAB_WARMUP_STEPS,
        help="PAB 开始生效的 step 索引（前几步做全量计算）",
    )
    parser.add_argument(
        "--pab_cooldown_steps_offset",
        type=int,
        default=PAB_COOLDOWN_STEPS_OFFSET,
        help="最后几步关闭 PAB（做全量计算），cooldown_steps = num_steps - offset",
    )
    parser.add_argument(
        "--pab_skip_range",
        type=int,
        default=PAB_SKIP_RANGE,
        help="PAB attention 复用间隔：每隔 skip_range 步重算一次，其余步复用缓存（越大越快，画质风险越高）",
    )
    parser.add_argument(
        "--disable_quality_eval",
        action="store_true",
        default=not quality_eval,
        help="关闭相对 ori 基准的质量评测（PSNR/SSIM/LPIPS 等）",
    )
    parser.add_argument(
        "--disable_progress_bar",
        action="store_true",
        default=False,
        help="关闭 diffusers 进度条",
    )
    return parser.parse_args()


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


def _resolve_output_paths(
    model_path: str,
    prompt: str,
    skip_range: int,
    output_dir: Path | None,
    output_image: Path | None,
) -> tuple[Path, Path, str]:
    """与 ori/teacache/cubic 一致：`{tag}_{prompt_id}_{model_id}.png`，便于 quality_metric 替换基准。"""
    model_id = Path(model_path).name
    prompt_id = get_prompt_id(prompt)
    speed_tag = f"skip{skip_range}"
    resolved_dir = output_dir or Path("results") / "firered" / model_id / TAG / f"{prompt_id}_{speed_tag}"
    resolved_name = f"{TAG}_{prompt_id}_{model_id}.png"
    resolved_image = output_image or (resolved_dir / resolved_name)
    return resolved_dir, resolved_image, model_id


def main() -> None:
    _init_logging()
    init_timer()
    args = parse_args()
    args.model_path = resolve_model_path(args.model_path, "--model_path")

    pab_cooldown_steps = args.num_inference_steps - args.pab_cooldown_steps_offset

    output_dir, output_image_path, model_id = _resolve_output_paths(
        args.model_path,
        args.prompt,
        args.pab_skip_range,
        args.output_dir,
        args.output_image,
    )
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading pipeline from {args.model_path}...", flush=True)
    pipeline = load_pipeline(args.model_path, args.disable_progress_bar)
    print("Pipeline loaded.", flush=True)

    transformer = pipeline.transformer
    transformer.enable_pab = True
    transformer.num_steps = args.num_inference_steps
    transformer.pab_warmup_steps = args.pab_warmup_steps
    transformer.pab_cooldown_steps = pab_cooldown_steps
    transformer.pab_skip_range = args.pab_skip_range
    if hasattr(transformer, "reset_pab_state"):
        transformer.reset_pab_state()

    logging.info(
        "[firered-pab] prompt_id=%s seed=%d steps=%d warmup=%d cooldown=%d skip_range=%d output_dir=%s",
        get_prompt_id(args.prompt),
        args.seed,
        args.num_inference_steps,
        args.pab_warmup_steps,
        pab_cooldown_steps,
        args.pab_skip_range,
        output_dir,
    )

    images = [Image.open(p).convert("RGB") for p in args.input_image]
    logging.info("[firered-pab] loaded_images=%d", len(images))

    inputs = {
        "image": images,
        "prompt": args.prompt,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": " ",
        "num_inference_steps": args.num_inference_steps,
        "num_images_per_prompt": 1,
    }

    with torch.inference_mode():
        result = pipeline(**inputs)

    output_image = result.images[0]
    output_image.save(output_image_path)

    quality_metrics = None
    if not args.disable_quality_eval:
        quality_result = evaluate_quality_with_origin(
            image_path=str(output_image_path),
            tag=TAG,
            save_metrics=False,
            baseline_path=None,
            output_dir=str(output_dir),
        )
        quality_metrics = quality_result["metrics"] if quality_result is not None else None

    print("Image saved at:", output_image_path.resolve(), flush=True)
    print_time_statistics(timer_names=REQUIRED_TIMER_NAMES)
    save_time_stats_to_file(
        str(output_dir / "timing_stats.txt"),
        timer_names=REQUIRED_TIMER_NAMES,
    )

    params = {
        "model": "firered",
        "method": TAG,
        "model_id": model_id,
        "model_path": args.model_path,
        "prompt": args.prompt,
        "input_image": [str(p) for p in args.input_image],
        "output_image": str(output_image_path),
        "seed": args.seed,
        "true_cfg_scale": args.true_cfg_scale,
        "num_inference_steps": args.num_inference_steps,
        "pab_warmup_steps": args.pab_warmup_steps,
        "pab_cooldown_steps": pab_cooldown_steps,
        "pab_skip_range": args.pab_skip_range,
        "quality_eval": not args.disable_quality_eval,
    }
    params_path = save_params_and_metrics(
        str(output_dir),
        TAG,
        params,
        get_time_statistics_dict(timer_names=REQUIRED_TIMER_NAMES),
        quality_metrics=quality_metrics,
    )
    print(f"Params & metrics saved to {params_path}", flush=True)


if __name__ == "__main__":
    main()
