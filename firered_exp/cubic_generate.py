"""FireRed Cubic monitor-only inference entry."""


import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image
from firered_cubic.pipeline import CubicQwenImageEditPlusPipeline
from firered_cubic.transformer import CubicQwenImageTransformer2DModel

from utils.launch_defaults import resolve_model_path
from utils.quality_metric import evaluate_quality_with_origin
from utils.results import get_prompt_id, save_params_and_metrics
from utils.timer import get_time_statistics_dict, init_timer, print_time_statistics, save_time_stats_to_file

IMAGE_PATH = "/home/fit/zhaijdcyy/WORK/Cubic/firered_exp/examples/cola.png"
IMAGE_PATH = "/home/chenyy/WORK/cyy/cubic/firered_exp/examples/cola.png"

PROMPT = "Transform the object into a realistic miniature product by carefully holding it between your thumb and forefinger."
REQUIRED_TIMER_NAMES = [
    "generate_e2e",
    "anchor_step_monitor",
    "anchor_step_partitioner",
    "anchor_step_optimizer",
    "anchor_step_total",
    "warmup_cooldown_step",
    "selective_forward",
]

SPEEDUP = 5
COND_SPEEDUP = 5
WARMUP_STEPS = 9
COOLDOWN_STEPS = 1

TAG = f"cubic_w{WARMUP_STEPS}_s{SPEEDUP}_cs{COND_SPEEDUP}"

visualize_curvature = True  # 是否启用曲率可视化（会增加额外计算开销）
quality_eval = True


def _build_step_latency_metrics(step_trace):
    trace = [dict(item) for item in step_trace]
    return {
        "cubic_step_trace": {
            "steps": trace,
            "text_lines": [item["text"] for item in trace],
        }
    }


def _merge_quality_and_step_metrics(quality_metrics, step_trace):
    merged = dict(quality_metrics) if isinstance(quality_metrics, dict) else {}
    merged.update(_build_step_latency_metrics(step_trace))
    return merged


def _print_step_latency_trace(step_trace):
    print("Cubic per-step latency / active tokens:", flush=True)
    for item in step_trace:
        print(item["text"], flush=True)


def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FireRed Cubic monitor-only inference script")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--input_image", type=Path, nargs="+", default=[Path(IMAGE_PATH)])
    parser.add_argument("--output_image", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--cubic_target_speedup", type=float, default=SPEEDUP)
    parser.add_argument("--cubic_cond_target_speedup", type=float, default=COND_SPEEDUP)
    parser.add_argument("--cubic_warmup_steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--cubic_cooldown_steps", type=int, default=COOLDOWN_STEPS)
    parser.add_argument("--cubic_warmup_fraction", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cubic_cooldown_fraction", type=float, default=None, help=argparse.SUPPRESS)
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
    args.model_path = resolve_model_path(args.model_path, "--model_path")
    if args.cubic_warmup_fraction is not None:
        args.cubic_warmup_steps = max(0, int(args.num_inference_steps * args.cubic_warmup_fraction))
    if args.cubic_cooldown_fraction is not None:
        args.cubic_cooldown_steps = max(0, int(args.num_inference_steps * args.cubic_cooldown_fraction))
    output_dir, output_image_path, model_id = _resolve_output_paths(
        args.model_path,
        args.prompt,
        args.cubic_target_speedup,
        args.output_dir,
        args.output_image,
    )
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading pipeline from {args.model_path}...", flush=True)
    pipeline = load_pipeline(args.model_path)
    print("Pipeline loaded.", flush=True)
    logging.info(
        "[firered-run] prompt_id=%s seed=%d steps=%d main_speedup=%.3f cond_speedup=%.3f warmup_steps=%d cooldown_steps=%d dynamic_anchor=2x_tau_max partition_mode=uniform visualization=%s output_dir=%s",
        get_prompt_id(args.prompt),
        args.seed,
        args.num_inference_steps,
        args.cubic_target_speedup,
        args.cubic_cond_target_speedup,
        args.cubic_warmup_steps,
        args.cubic_cooldown_steps,
        not args.disable_visualization,
        output_dir,
    )

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
        "cubic_warmup_steps": args.cubic_warmup_steps,
        "cubic_cooldown_steps": args.cubic_cooldown_steps,
        "cubic_enable_visualization": not args.disable_visualization,
        "cubic_results_dir": str(output_dir),
        "cubic_visual_tag": f"seed{args.seed}",
    }

    with torch.inference_mode():
        result = pipeline(**inputs)
    step_trace = pipeline.get_step_trace()

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

    print("Image saved at:", output_image_path.resolve())
    _print_step_latency_trace(step_trace)
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
        "target_speedup": args.cubic_target_speedup,
        "cond_target_speedup": args.cubic_cond_target_speedup,
        "warmup_steps": args.cubic_warmup_steps,
        "cooldown_steps": args.cubic_cooldown_steps,
        "dynamic_anchor": "2x_tau_max",
        "partition_mode": "uniform",
        "visualization_enabled": not args.disable_visualization,
        "quality_eval": not args.disable_quality_eval,
    }
    params_path = save_params_and_metrics(
        str(output_dir),
        TAG,
        params,
        get_time_statistics_dict(timer_names=REQUIRED_TIMER_NAMES),
        quality_metrics=_merge_quality_and_step_metrics(quality_metrics, step_trace),
    )
    print(f"Params & metrics saved to {params_path}", flush=True)


if __name__ == "__main__":
    main()
