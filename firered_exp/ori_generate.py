"""FireRed-Image-Edit Inference Demo."""


import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from firered.pipeline import QwenImageEditPlusPipeline
from firered.transformer import QwenImageTransformer2DModel

from utils.results import get_prompt_id, save_params_and_metrics
from utils.timer import get_time_statistics_dict, get_timer, init_timer, print_time_statistics

MODEL_PATH = "/home/fit/zhaijdcyy/WORK/models/FireRed-Image-Edit-1.0"
IMAGE_PATH = "/home/fit/zhaijdcyy/WORK/Cubic/firered_exp/examples/cola.png"
PROMPT = "Transform the object into a realistic miniature product by carefully holding it between your thumb and forefinger."
TAG = "ori"

init_timer()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FireRed-Image-Edit inference script"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help="Path to the model or HuggingFace model ID",
    )
    parser.add_argument(
        "--input_image",
        type=Path,
        nargs="+",
        default=[Path(IMAGE_PATH)],
        help="Path(s) to the input image(s). Supports 1-N images. "
             "When more than 3 images are given the agent will "
             "automatically crop and stitch them into 2-3 composites.",
    )
    parser.add_argument(
        "--output_image",
        type=Path,
        default=None,
        help="Path to save the output image",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. If not set, uses results/firered/{model_id}/ori/{prompt_id}",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=PROMPT,
        help="Editing prompt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=49,
        help="Random seed for generation",
    )
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=4.0,
        help="True CFG scale",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=40,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--recaption",
        action="store_true",
        default=False,
        help="Enable agent-based recaption: expand the editing prompt to "
             "~512 words/characters via Gemini for richer context. "
             "Requires GEMINI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--optimized",
        action="store_true",
        default=False,
        help="Enable Int8, Cache, and Compile",
    )
    return parser.parse_args()



def load_pipeline(model_path: str) -> QwenImageEditPlusPipeline:
    """Load FireRed image edit pipeline."""
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
    pipe.set_progress_bar_config(disable=None)
    return pipe


def _resolve_output_paths(model_path: str, prompt: str, output_dir: Path | None, output_image: Path | None) -> tuple[Path, Path, str]:
    model_id = Path(model_path).name
    prompt_id = get_prompt_id(prompt)
    resolved_dir = output_dir or Path("results") / "firered" / model_id / TAG / prompt_id
    resolved_name = f"{TAG}_{prompt_id}_{model_id}.png"
    resolved_image = output_image or (resolved_dir / resolved_name)
    return resolved_dir, resolved_image, model_id



def main() -> None:
    """Main entry point."""
    args = parse_args()
    output_dir, output_image_path, model_id = _resolve_output_paths(
        args.model_path, args.prompt, args.output_dir, args.output_image
    )
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading pipeline from {args.model_path}...", flush=True)
    with get_timer("load_pipeline"):
        pipeline = load_pipeline(args.model_path)
    print("Pipeline loaded.", flush=True)

    # ── Load all input images ──
    with get_timer("load_images"):
        images = [Image.open(p).convert("RGB") for p in args.input_image]
    prompt = args.prompt
    print(f"Loaded {len(images)} image(s).")


    # ── Agent: stitch + recaption when needed ──
    # need_stitch = len(images) > 3
    # need_recaption = args.recaption


    # if need_stitch or need_recaption:
    #     from agent import AgentPipeline


    #     agent = AgentPipeline(verbose=True)
    #     agent_result = agent.run(
    #         images,
    #         prompt,
    #         enable_recaption=need_recaption or need_stitch,
    #     )
    #     images = agent_result.images
    #     prompt = agent_result.prompt
    #     print(f"Agent produced {len(images)} image(s).")
    #     print(f"Rewritten prompt: {prompt[:200]}{'…' if len(prompt) > 200 else ''}")


    inputs = {
        "image": images,
        "prompt": prompt,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": " ",
        "num_inference_steps": args.num_inference_steps,
        "num_images_per_prompt": 1,
    }


    if args.optimized:
        print("NOTE: The first inference after compilation may take 1-2 minutes.")

    with get_timer("inference_first"):
        with torch.inference_mode():
            result = pipeline(**inputs)

    output_image = result.images[0]
    with get_timer("save_image"):
        output_image.save(output_image_path)

    print("Image saved at:", output_image_path.resolve())

    if args.optimized:
        print("Subsequent runs will be significantly faster. Enjoy~")
        with get_timer("inference_second"):
            with torch.inference_mode():
                result = pipeline(**inputs)
        output_image = result.images[0]
        with get_timer("save_image_second"):
            output_image.save(output_image_path)

    print_time_statistics()

    params = {
        "model": "firered",
        "method": TAG,
        "model_id": model_id,
        "model_path": args.model_path,
        "prompt": prompt,
        "input_image": [str(p) for p in args.input_image],
        "output_image": str(output_image_path),
        "seed": args.seed,
        "true_cfg_scale": args.true_cfg_scale,
        "num_inference_steps": args.num_inference_steps,
        "optimized": args.optimized,
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