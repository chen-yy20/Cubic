import argparse
import os

import torch

from flux.pipeline import FluxPipeline
from flux.transformer import FluxTransformer2DModel
from utils.launch_defaults import resolve_model_path
from utils.results import get_prompt_id, save_params_and_metrics
from utils.timer import (
    disable_timing,
    enable_timing,
    get_time_statistics_dict,
    init_timer,
    print_time_statistics,
)

HEIGHT = 1024
WIDTH = 1024
PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 3.5
TAG = "ori"
NUM_IMAGES = 3
BASE_SEED = 42
WARMUP_RUNS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flux inference script")
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
    return parser.parse_args()


def load_pipeline(model_path: str) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.transformer = FluxTransformer2DModel.from_pretrained(
        f"{model_path}/transformer",
        torch_dtype=torch.bfloat16,
    )
    return pipe.to("cuda")


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_path, "--model_path")
    model_id = os.path.basename(model_path.rstrip("/"))
    prompt_id = get_prompt_id(args.prompt)
    output_dir = args.output_dir or f"./results/flux/{TAG}/{prompt_id}"
    seeds = [args.seed + i for i in range(max(1, int(args.num_images)))]

    init_timer()
    os.makedirs(output_dir, exist_ok=True)

    pipe = load_pipeline(model_path)
    print(f"Model loaded, GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB", flush=True)

    disable_timing()
    warmup_generator = torch.Generator("cuda").manual_seed(seeds[0])
    for _ in range(max(0, int(args.warmup_runs))):
        with torch.inference_mode():
            pipe(
                args.prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                max_sequence_length=512,
                generator=warmup_generator,
            )

    enable_timing()
    for seed in seeds:
        generator = torch.Generator("cuda").manual_seed(seed)
        with torch.inference_mode():
            image = pipe(
                args.prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                max_sequence_length=512,
                generator=generator,
            ).images[0]
        output_path = os.path.join(output_dir, f"{TAG}_{prompt_id}_seed{seed}.png")
        image.save(output_path)
        print(f"Stored {output_path}!", flush=True)

    print_time_statistics()
    params = {
        "model": "flux",
        "method": TAG,
        "model_id": model_id,
        "model_path": model_path,
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_images": len(seeds),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seeds": seeds,
        "warmup_runs": max(0, int(args.warmup_runs)),
    }
    params_path = save_params_and_metrics(
        output_dir,
        TAG,
        params,
        get_time_statistics_dict(),
        quality_metrics=None,
    )
    print(f"Params & metrics saved to {params_path}", flush=True)


if __name__ == "__main__":
    main()
