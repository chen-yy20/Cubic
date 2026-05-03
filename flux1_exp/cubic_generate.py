import argparse
import os
from statistics import mean

import torch

from flux.transformer import FluxTransformer2DModel
from flux_cubic.pipeline import CubicFluxPipeline
from utils.launch_defaults import resolve_model_path
from utils.quality_metric import evaluate_quality_with_origin
from utils.results import get_prompt_id, save_params_and_metrics
from utils.timer import get_time_statistics_dict, init_timer, print_time_statistics, save_time_stats_to_file

HEIGHT = 1024
WIDTH = 1024
PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 3.5
NUM_IMAGES = 3
BASE_SEED = 42
WARMUP_RUNS = 0

SPEEDUP = 5
WARMUP_STEPS = 9
COOLDOWN_STEPS = 1


curvature_visualize = False # Set to True to enable curvature visualization, which will store curvature maps and token selection maps for each step in the specified results directory. This is useful for analyzing the behavior of the cubic scheduling, but will consume more storage and slightly more time due to the additional I/O operations.
quality_eval = True

TAG = f"cubic_w{WARMUP_STEPS}_s{SPEEDUP}"
REQUIRED_TIMER_NAMES = [
    "generate_e2e",
    "anchor_step_monitor",
    "anchor_step_partitioner",
    "anchor_step_optimizer",
    "anchor_step_total",
    "warmup_cooldown_step",
    "selective_forward",
]

def _build_step_latency_metrics(step_trace, per_seed_trace=None):
    trace = [dict(item) for item in step_trace]
    payload = {
        "steps": trace,
        "text_lines": [item["text"] for item in trace],
    }
    if per_seed_trace is not None:
        payload["per_seed"] = {
            seed_key: {
                "steps": [dict(item) for item in seed_trace],
                "text_lines": [item["text"] for item in seed_trace],
            }
            for seed_key, seed_trace in per_seed_trace.items()
        }
    return {"cubic_step_trace": payload}


def _merge_quality_and_step_metrics(quality_metrics, step_trace, per_seed_trace=None):
    merged = dict(quality_metrics) if isinstance(quality_metrics, dict) else {}
    merged.update(_build_step_latency_metrics(step_trace, per_seed_trace=per_seed_trace))
    return merged


def _print_step_latency_trace(step_trace, seed=None):
    prefix = "Cubic per-step latency / active tokens"
    if seed is not None:
        prefix += f" (seed={seed})"
    print(f"{prefix}:", flush=True)
    for item in step_trace:
        print(item["text"], flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flux cubic inference script")
    parser.add_argument("--model_path", type=str, default=None, help="Model path or HuggingFace model ID")
    parser.add_argument("--prompt", type=str, default=PROMPT, help="Generation prompt")
    parser.add_argument("--height", type=int, default=HEIGHT, help="Output image height")
    parser.add_argument("--width", type=int, default=WIDTH, help="Output image width")
    parser.add_argument("--num_inference_steps", type=int, default=NUM_INFERENCE_STEPS, help="Inference steps")
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Base random seed")
    parser.add_argument("--num_images", type=int, default=NUM_IMAGES, help="Number of images/seeds to generate")
    parser.add_argument("--warmup_runs", type=int, default=WARMUP_RUNS, help="Warmup runs before timed generation")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory override")
    parser.add_argument("--cubic_target_speedup", type=float, default=SPEEDUP, help="Target cubic speedup")
    parser.add_argument("--cubic_warmup_steps", type=int, default=WARMUP_STEPS, help="Warmup steps")
    parser.add_argument("--cubic_cooldown_steps", type=int, default=COOLDOWN_STEPS, help="Cooldown steps")
    parser.add_argument(
        "--disable_visualization",
        action="store_true",
        default=not curvature_visualize,
        help="Disable curvature visualization export",
    )
    parser.add_argument(
        "--disable_quality_eval",
        action="store_true",
        default=not quality_eval,
        help="Disable quality evaluation against ori baseline",
    )
    return parser.parse_args()


def load_pipeline(model_path: str) -> CubicFluxPipeline:
    pipe = CubicFluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.transformer = FluxTransformer2DModel.from_pretrained(
        f"{model_path}/transformer",
        torch_dtype=torch.bfloat16,
    )
    return pipe.to("cuda")


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_path, "--model_path")
    model_id = os.path.basename(model_path.rstrip("/"))
    tag = f"cubic_w{args.cubic_warmup_steps}_s{args.cubic_target_speedup:g}"
    prompt_id = get_prompt_id(args.prompt)
    output_dir = args.output_dir or f"./results/flux/{tag}/{prompt_id}_{args.cubic_target_speedup:g}x"
    seeds = [args.seed + i for i in range(max(1, int(args.num_images)))]

    init_timer()
    os.makedirs(output_dir, exist_ok=True)

    pipe = load_pipeline(model_path)
    print(f"Model loaded, GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB", flush=True)

    per_seed_quality: dict[str, dict] = {}
    per_seed_step_trace: dict[str, list[dict]] = {}

    for warmup_idx in range(max(0, int(args.warmup_runs))):
        warmup_generator = torch.Generator("cuda").manual_seed(seeds[0] + warmup_idx)
        with torch.inference_mode():
            pipe(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                max_sequence_length=512,
                generator=warmup_generator,
                cubic_warmup_steps=args.cubic_warmup_steps,
                cubic_cooldown_steps=args.cubic_cooldown_steps,
                cubic_target_speedup=args.cubic_target_speedup,
                cubic_enable_visualization=not args.disable_visualization,
                cubic_results_dir=output_dir,
                cubic_visual_tag=f"warmup{warmup_idx}",
            )

    for seed in seeds:
        generator = torch.Generator("cuda").manual_seed(seed)
        with torch.inference_mode():
            image = pipe(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                max_sequence_length=512,
                generator=generator,
                cubic_warmup_steps=args.cubic_warmup_steps,
                cubic_cooldown_steps=args.cubic_cooldown_steps,
                cubic_target_speedup=args.cubic_target_speedup,
                cubic_enable_visualization=not args.disable_visualization,
                cubic_results_dir=output_dir,
                cubic_visual_tag=f"seed{seed}",
            ).images[0]
        step_trace = pipe.get_step_trace()
        per_seed_step_trace[f"seed{seed}"] = step_trace
        _print_step_latency_trace(step_trace, seed=seed)
        output_path = os.path.join(output_dir, f"{tag}_{prompt_id}_seed{seed}.png")
        image.save(output_path)
        print(f"Stored {output_path}!", flush=True)

        if not args.disable_quality_eval:
            quality_result = evaluate_quality_with_origin(
                image_path=output_path,
                tag=tag,
                save_metrics=False,
                baseline_path=None,
                output_dir=output_dir,
            )
            per_seed_quality[f"seed{seed}"] = quality_result["metrics"] if quality_result is not None else {}

    quality_metrics = None
    if not args.disable_quality_eval:
        valid_metrics = [m for m in per_seed_quality.values() if m]
        if valid_metrics:
            metric_names = ["psnr", "ssim", "lpips", "fid", "fvd"]
            mean_metrics = {}
            for name in metric_names:
                values = [float(m[name]) for m in valid_metrics if m.get(name) is not None]
                mean_metrics[name] = mean(values) if values else None
            quality_metrics = {
                "per_seed": per_seed_quality,
                "mean": mean_metrics,
            }
        else:
            quality_metrics = {
                "per_seed": per_seed_quality,
                "mean": None,
            }

    print_time_statistics(timer_names=REQUIRED_TIMER_NAMES)
    save_time_stats_to_file(
        os.path.join(output_dir, "timing_stats.txt"),
        timer_names=REQUIRED_TIMER_NAMES,
    )
    first_step_trace = next(iter(per_seed_step_trace.values()), [])
    params = {
        "model": "flux",
        "method": tag,
        "model_id": model_id,
        "model_path": model_path,
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_images": len(seeds),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seeds": seeds,
        "target_speedup": args.cubic_target_speedup,
        "warmup_steps": args.cubic_warmup_steps,
        "cooldown_steps": args.cubic_cooldown_steps,
        "dynamic_anchor": "2x_tau_max",
        "partition_mode": "uniform",
        "uniform_common_factor_blocks": True,
        "quality_eval": not args.disable_quality_eval,
        "visualization_enabled": not args.disable_visualization,
        "warmup_runs": max(0, int(args.warmup_runs)),
    }
    params_path = save_params_and_metrics(
        output_dir,
        tag,
        params,
        get_time_statistics_dict(timer_names=REQUIRED_TIMER_NAMES),
        quality_metrics=_merge_quality_and_step_metrics(
            quality_metrics,
            first_step_trace,
            per_seed_trace=per_seed_step_trace,
        ),
    )
    print(f"Params & metrics saved to {params_path}", flush=True)


if __name__ == "__main__":
    main()
