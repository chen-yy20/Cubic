"""Flux2-klein inference demo with custom pipeline/transformer loading."""

import argparse
from pathlib import Path

import torch

from flux2_klein.pipeline import Flux2KleinPipeline
from flux2_klein.transformer import Flux2Transformer2DModel
from utils.results import get_prompt_id, save_params_and_metrics
from utils.timer import get_timer, get_time_statistics_dict, init_timer, print_time_statistics

MODEL_PATH = "/home/chenyy/WORK/models/flux2-klein"  # black-forest-labs/FLUX.2-klein-4B
PROMPT = "A cat holding a sign that says hello world"
TAG = "ori"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flux2-klein inference script")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Model path or HuggingFace model ID")
    parser.add_argument("--prompt", type=str, default=PROMPT, help="Generation prompt")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. If not set, uses results/flux2/{model_id}/ori/{prompt_id}",
    )
    parser.add_argument(
        "--output_image",
        type=Path,
        default=None,
        help="Output image path. If not set, uses {output_dir}/{tag}_{prompt_id}_{model_id}.png",
    )
    parser.add_argument("--height", type=int, default=1024, help="Output image height")
    parser.add_argument("--width", type=int, default=1024, help="Output image width")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Inference steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device")
    return parser.parse_args()


def _resolve_output_paths(
    model_path: str,
    prompt: str,
    output_dir: Path | None,
    output_image: Path | None,
) -> tuple[Path, Path, str]:
    model_id = Path(model_path).name
    prompt_id = get_prompt_id(prompt)
    resolved_dir = output_dir or Path("results") / "flux2" / model_id / TAG / prompt_id
    resolved_name = f"{TAG}_{prompt_id}_{model_id}.png"
    resolved_image = output_image or (resolved_dir / resolved_name)
    return resolved_dir, resolved_image, model_id


def load_pipeline(model_path: str, dtype: torch.dtype, device: str) -> Flux2KleinPipeline:
    print(f"Loading custom Flux2Klein pipeline from {model_path}...", flush=True)
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    pipe = Flux2KleinPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        transformer=transformer,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    print(f"Pipeline ready on device={pipe._execution_device}", flush=True)
    return pipe


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16
    output_dir, output_image_path, model_id = _resolve_output_paths(
        args.model_path, args.prompt, args.output_dir, args.output_image
    )

    init_timer()

    with get_timer("load_pipeline"):
        pipe = load_pipeline(args.model_path, dtype, args.device)

    generator_device = pipe._execution_device.type
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)
    print(
        "Inference config: "
        f"height={args.height}, width={args.width}, steps={args.num_inference_steps}, "
        f"guidance_scale={args.guidance_scale}, seed={args.seed}, generator_device={generator_device}",
        flush=True,
    )

    with get_timer("inference"):
        image = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).images[0]

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    with get_timer("save_image"):
        image.save(output_image_path)

    print(f"Image saved to: {output_image_path.resolve()}", flush=True)
    print_time_statistics()

    params = {
        "model": "flux2",
        "method": TAG,
        "model_id": model_id,
        "model_path": args.model_path,
        "prompt": args.prompt,
        "output_image": str(output_image_path),
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "device": args.device,
    }
    params_path = save_params_and_metrics(
        str(output_dir),
        TAG,
        params,
        get_time_statistics_dict(),
        quality_metrics=None,
    )
    print(f"Params & metrics saved to {params_path}", flush=True)


if __name__ == "__main__":
    main()
