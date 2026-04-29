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
SEEDS = [42 + i for i in range(NUM_IMAGES)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flux inference script")
    parser.add_argument("--model_path", type=str, default=None, help="Model path or HuggingFace model ID")
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
    output_dir = f"./results/flux/{TAG}/{get_prompt_id(PROMPT)}"

    init_timer()
    os.makedirs(output_dir, exist_ok=True)

    prompt = PROMPT
    pipe = load_pipeline(model_path)
    print(f"Model loaded, GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB", flush=True)

    disable_timing()
    warmup_generator = torch.Generator("cuda").manual_seed(SEEDS[0])
    for _ in range(1):
        with torch.inference_mode():
            pipe(
                prompt,
                height=HEIGHT,
                width=WIDTH,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                max_sequence_length=512,
                generator=warmup_generator,
            )

    enable_timing()
    for seed in SEEDS:
        generator = torch.Generator("cuda").manual_seed(seed)
        with torch.inference_mode():
            image = pipe(
                prompt,
                height=HEIGHT,
                width=WIDTH,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                max_sequence_length=512,
                generator=generator,
            ).images[0]
        output_path = os.path.join(output_dir, f"{TAG}_{get_prompt_id(PROMPT)}_seed{seed}.png")
        image.save(output_path)
        print(f"Stored {output_path}!", flush=True)

    print_time_statistics()
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
