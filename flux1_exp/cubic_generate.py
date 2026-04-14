import os

import torch

from flux.transformer import FluxTransformer2DModel
from flux_cubic.pipeline import CubicFluxPipeline
from utils.results import get_prompt_id, save_params_and_metrics
from utils.timer import get_time_statistics_dict, init_timer, print_time_statistics

HEIGHT = 1024
WIDTH = 1024
MODEL_PATH = "/home/chenyy/WORK/models/Flux-1"
PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 3.5
NUM_IMAGES = 3
SEEDS = [42 + i for i in range(NUM_IMAGES)]

SPEEDUP = 3
WARMUP_FRACTION = 0.15
COOLDOWN_FRACTION = 0.15
ANCHOR_INTERVAL = 10

TAG = "cubic"
OUTPUT_DIR = f"./results/flux/{TAG}/{get_prompt_id(PROMPT)}_{SPEEDUP}x"
MODEL_ID = os.path.basename(MODEL_PATH.rstrip("/"))

curvature_visualize = False # Set to True to enable curvature visualization, which will store curvature maps and token selection maps for each step in the specified results directory. This is useful for analyzing the behavior of the cubic scheduling, but will consume more storage and slightly more time due to the additional I/O operations.


def load_pipeline(model_path: str) -> CubicFluxPipeline:
    pipe = CubicFluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.transformer = FluxTransformer2DModel.from_pretrained(
        f"{model_path}/transformer",
        torch_dtype=torch.bfloat16,
    )
    return pipe.to("cuda")


def main() -> None:
    init_timer()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipe = load_pipeline(MODEL_PATH)
    print(f"Model loaded, GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB", flush=True)

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
                cubic_warmup_fraction=WARMUP_FRACTION,
                cubic_cooldown_fraction=COOLDOWN_FRACTION,
                cubic_anchor_interval=ANCHOR_INTERVAL,
                cubic_target_speedup=SPEEDUP,
                cubic_enable_visualization=curvature_visualize,
                cubic_results_dir=OUTPUT_DIR,
                cubic_visual_tag=f"seed{seed}",
            ).images[0]
        output_path = os.path.join(OUTPUT_DIR, f"{TAG}_{get_prompt_id(PROMPT)}_seed{seed}.png")
        image.save(output_path)
        print(f"Stored {output_path}!", flush=True)

    print_time_statistics()
    params = {
        "model": "flux",
        "method": TAG,
        "model_id": MODEL_ID,
        "model_path": MODEL_PATH,
        "prompt": PROMPT,
        "height": HEIGHT,
        "width": WIDTH,
        "num_images": NUM_IMAGES,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "seeds": SEEDS,
        "target_speedup": SPEEDUP,
        "warmup_fraction": WARMUP_FRACTION,
        "cooldown_fraction": COOLDOWN_FRACTION,
        "anchor_interval": ANCHOR_INTERVAL,
    }
    params_path = save_params_and_metrics(
        OUTPUT_DIR,
        TAG,
        params,
        get_time_statistics_dict(),
        quality_metrics=None,
    )
    print(f"Params & metrics saved to {params_path}", flush=True)


if __name__ == "__main__":
    main()
