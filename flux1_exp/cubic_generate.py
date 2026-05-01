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
SEEDS = [42 + i for i in range(NUM_IMAGES)]

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
    output_dir = f"./results/flux/{TAG}/{get_prompt_id(PROMPT)}_{SPEEDUP}x"

    init_timer()
    os.makedirs(output_dir, exist_ok=True)

    pipe = load_pipeline(model_path)
    print(f"Model loaded, GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB", flush=True)

    per_seed_quality: dict[str, dict] = {}
    per_seed_step_trace: dict[str, list[dict]] = {}

    for seed in SEEDS:
        generator = torch.Generator("cuda").manual_seed(seed)
        with torch.inference_mode():
            image = pipe(
                prompt=PROMPT,
                height=HEIGHT,
                width=WIDTH,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                max_sequence_length=512,
                generator=generator,
                cubic_warmup_steps=WARMUP_STEPS,
                cubic_cooldown_steps=COOLDOWN_STEPS,
                cubic_target_speedup=SPEEDUP,
                cubic_enable_visualization=curvature_visualize,
                cubic_results_dir=output_dir,
                cubic_visual_tag=f"seed{seed}",
            ).images[0]
        step_trace = pipe.get_step_trace()
        per_seed_step_trace[f"seed{seed}"] = step_trace
        _print_step_latency_trace(step_trace, seed=seed)
        output_path = os.path.join(output_dir, f"{TAG}_{get_prompt_id(PROMPT)}_seed{seed}.png")
        image.save(output_path)
        print(f"Stored {output_path}!", flush=True)

        if quality_eval:
            quality_result = evaluate_quality_with_origin(
                image_path=output_path,
                tag=TAG,
                save_metrics=False,
                baseline_path=None,
                output_dir=output_dir,
            )
            per_seed_quality[f"seed{seed}"] = quality_result["metrics"] if quality_result is not None else {}

    quality_metrics = None
    if quality_eval:
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
        "method": TAG,
        "model_id": model_id,
        "model_path": model_path,
        "prompt": PROMPT,
        "height": HEIGHT,
        "width": WIDTH,
        "num_images": NUM_IMAGES,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "seeds": SEEDS,
        "target_speedup": SPEEDUP,
        "warmup_steps": WARMUP_STEPS,
        "cooldown_steps": COOLDOWN_STEPS,
        "dynamic_anchor": "2x_tau_max",
        "partition_mode": "uniform",
        "uniform_common_factor_blocks": True,
        "quality_eval": quality_eval,
    }
    params_path = save_params_and_metrics(
        output_dir,
        TAG,
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
