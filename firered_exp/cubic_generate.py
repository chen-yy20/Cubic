"""FireRed Cubic monitor-only inference entry."""


import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image
from firered_cubic.pipeline import CubicQwenImageEditPlusPipeline
from firered_cubic.transformer import CubicQwenImageTransformer2DModel

from utils.quality_metric import evaluate_quality_with_origin
from utils.results import get_prompt_id, save_params_and_metrics
from utils.timer import get_time_statistics_dict, get_timer, init_timer, print_time_statistics

MODEL_PATH = "/home/fit/zhaijdcyy/WORK/models/FireRed-Image-Edit-1.0"
IMAGE_PATH = "/home/fit/zhaijdcyy/WORK/Cubic/firered_exp/examples/cola.png"
PROMPT = "Transform the object into a realistic miniature product by carefully holding it between your thumb and forefinger."
TAG = "cubic"

SPEEDUP = 2.0
COND_SPEEDUP = 2.0
WARMUP_FRACTION = 0.1
COOLDOWN_FRACTION = 0.1
ANCHOR_INTERVAL = 8

visualize_curvature = False  # 是否启用曲率可视化（会增加额外计算开销）
quality_eval = True


def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FireRed Cubic monitor-only inference script")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--input_image", type=Path, nargs="+", default=[Path(IMAGE_PATH)])
    parser.add_argument("--output_image", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--cubic_target_speedup", type=float, default=SPEEDUP)
    parser.add_argument("--cubic_cond_target_speedup", type=float, default=COND_SPEEDUP)
    parser.add_argument("--cubic_warmup_fraction", type=float, default=WARMUP_FRACTION)
    parser.add_argument("--cubic_cooldown_fraction", type=float, default=COOLDOWN_FRACTION)
    parser.add_argument("--cubic_anchor_interval", type=int, default=ANCHOR_INTERVAL)
    parser.add_argument("--disable_visualization", action="store_true", default=not visualize_curvature, help="Disable cubic curvature visualization export")
    parser.add_argument("--disable_quality_eval", action="store_true", default=not quality_eval, help="Disable quality evaluation against origin baseline")
    return parser.parse_args()


def load_pipeline(model_path: str) -> CubicQwenImageEditPlusPipeline:
    pipe = CubicQwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        transformer=CubicQwenImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        ),
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=None)
    return pipe


def _resolve_output_paths(
    model_path: str,
    prompt: str,
    target_speedup: float,
    output_dir: Path | None,
    output_image: Path | None,
) -> tuple[Path, Path, str]:
    model_id = Path(model_path).name
    prompt_id = get_prompt_id(prompt)
    speed_tag = f"{target_speedup:g}x"
    resolved_dir = output_dir or Path("results") / "firered" / model_id / TAG / f"{prompt_id}_{speed_tag}"
    resolved_name = f"{TAG}_{prompt_id}_{model_id}.png"
    resolved_image = output_image or (resolved_dir / resolved_name)
    return resolved_dir, resolved_image, model_id


def main() -> None:
    _init_logging()
    init_timer()
    args = parse_args()
    output_dir, output_image_path, model_id = _resolve_output_paths(
        args.model_path,
        args.prompt,
        args.cubic_target_speedup,
        args.output_dir,
        args.output_image,
    )
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading pipeline from {args.model_path}...", flush=True)
    with get_timer("load_pipeline"):
        pipeline = load_pipeline(args.model_path)
    print("Pipeline loaded.", flush=True)
    logging.info(
        "[firered-run] prompt_id=%s seed=%d steps=%d main_speedup=%.3f cond_speedup=%.3f warmup_fraction=%.3f cooldown_fraction=%.3f anchor_interval=%d visualization=%s output_dir=%s",
        get_prompt_id(args.prompt),
        args.seed,
        args.num_inference_steps,
        args.cubic_target_speedup,
        args.cubic_cond_target_speedup,
        args.cubic_warmup_fraction,
        args.cubic_cooldown_fraction,
        args.cubic_anchor_interval,
        not args.disable_visualization,
        output_dir,
    )

    with get_timer("load_images"):
        images = [Image.open(p).convert("RGB") for p in args.input_image]
    logging.info("[firered-run] loaded_images=%d", len(images))

    inputs = {
        "image": images,
        "prompt": args.prompt,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": " ",
        "num_inference_steps": args.num_inference_steps,
        "num_images_per_prompt": 1,
        "cubic_target_speedup": args.cubic_target_speedup,
        "cubic_cond_target_speedup": args.cubic_cond_target_speedup,
        "cubic_warmup_fraction": args.cubic_warmup_fraction,
        "cubic_cooldown_fraction": args.cubic_cooldown_fraction,
        "cubic_anchor_interval": args.cubic_anchor_interval,
        "cubic_enable_visualization": not args.disable_visualization,
        "cubic_results_dir": str(output_dir),
        "cubic_visual_tag": f"seed{args.seed}",
    }

    with get_timer("inference_first"):
        with torch.inference_mode():
            result = pipeline(**inputs)

    output_image = result.images[0]
    with get_timer("save_image"):
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

    print("Image saved at:", output_image_path.resolve())
    print_time_statistics()

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
        "target_speedup": args.cubic_target_speedup,
        "cond_target_speedup": args.cubic_cond_target_speedup,
        "warmup_fraction": args.cubic_warmup_fraction,
        "cooldown_fraction": args.cubic_cooldown_fraction,
        "anchor_interval": args.cubic_anchor_interval,
        "visualization_enabled": not args.disable_visualization,
        "quality_eval": not args.disable_quality_eval,
    }
    params_path = save_params_and_metrics(
        str(output_dir),
        TAG,
        params,
        get_time_statistics_dict(),
        quality_metrics=quality_metrics,
    )
    print(f"Params & metrics saved to {params_path}", flush=True)


if __name__ == "__main__":
    main()
