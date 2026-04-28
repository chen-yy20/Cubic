"""Flux + TeaCache 推理入口。

TeaCache 论文：https://arxiv.org/abs/2411.19108
实现：baseline/teacache_flux.py（monkey patch FluxTransformer2DModel.forward）

用法（从 Cubic 根目录）：
    python launch.py --model flux --method teacache [--launcher python]
    python launch.py --model flux --method teacache -- --rel_l1_thresh 0.5
"""

from __future__ import annotations

import argparse
import os
from statistics import mean

import torch

from baseline.teacache_flux import reset_teacache_state, teacache_forward

from flux.pipeline import FluxPipeline
from flux.transformer import FluxTransformer2DModel

# teacache_flux.py 默认 patch 的是 diffusers 的 FluxTransformer2DModel，
# 但本项目 flux/transformer.py 定义了自己的同名类，需要单独 patch。
FluxTransformer2DModel.forward = teacache_forward
from utils.quality_metric import evaluate_quality_with_origin
from utils.results import get_prompt_id, save_params_and_metrics
from utils.timer import (
    get_time_statistics_dict,
    get_timer,
    init_timer,
    print_time_statistics,
)

# ── 默认配置（与 ori/cubic 保持一致，便于质量对比）────────────────────────────
MODEL_PATH         = "/home/fit/zhaijdcyy/WORK/models/Flux-1"
PROMPT             = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."
HEIGHT             = 1024
WIDTH              = 1024
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE     = 3.5
NUM_IMAGES         = 3
SEEDS              = [42 + i for i in range(NUM_IMAGES)]
REL_L1_THRESH      = 0.0   # 官方默认；越大越快质量损失越多（0.25/0.4/0.6/0.8 对应 ~12/10/9/8s）

TAG      = "teacache"
MODEL_ID = os.path.basename(MODEL_PATH.rstrip("/"))

quality_eval = True


# ── 参数解析 ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flux TeaCache inference script")
    parser.add_argument("--model_path",          type=str,   default=MODEL_PATH)
    parser.add_argument("--prompt",              type=str,   default=PROMPT)
    parser.add_argument("--height",              type=int,   default=HEIGHT)
    parser.add_argument("--width",               type=int,   default=WIDTH)
    parser.add_argument("--num_inference_steps", type=int,   default=NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance_scale",      type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--num_images",          type=int,   default=NUM_IMAGES)
    parser.add_argument("--seeds",               type=int,   nargs="+", default=None,
                        help="生成用的随机种子列表，默认 42,43,44,…")
    parser.add_argument("--rel_l1_thresh",       type=float, default=REL_L1_THRESH,
                        help="TeaCache 累积阈值；越大跳步越多越快，画质损失越大"
                             "（官方推荐 0.25/0.4/0.6/0.8，对应 ~12/10/9/8s）")
    parser.add_argument("--output_dir",          type=str,   default=None,
                        help="结果输出目录，默认 ./results/flux/teacache/{prompt_id}")
    parser.add_argument("--disable_quality_eval", action="store_true", default=not quality_eval,
                        help="关闭相对 ori 基准的质量评测（PSNR/SSIM/LPIPS 等）")
    return parser.parse_args()


# ── Pipeline 加载 ─────────────────────────────────────────────────────────────

def load_pipeline(model_path: str) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.transformer = FluxTransformer2DModel.from_pretrained(
        f"{model_path}/transformer",
        torch_dtype=torch.bfloat16,
    )
    return pipe.to("cuda")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    init_timer()
    args = parse_args()

    seeds = args.seeds if args.seeds is not None else [42 + i for i in range(args.num_images)]
    prompt_id  = get_prompt_id(args.prompt)
    model_id   = os.path.basename(args.model_path.rstrip("/"))
    output_dir = args.output_dir or os.path.join("results", "flux", TAG, prompt_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[teacache-flux] Loading pipeline from {args.model_path}...", flush=True)
    with get_timer("load_pipeline"):
        pipe = load_pipeline(args.model_path)
    print(
        f"[teacache-flux] Pipeline loaded. "
        f"GPU memory: {torch.cuda.memory_allocated() / 1024**2:.0f} MB",
        flush=True,
    )

    # 配置 TeaCache 超参
    transformer: FluxTransformer2DModel = pipe.transformer
    transformer.enable_teacache = True
    transformer.num_steps        = args.num_inference_steps
    transformer.rel_l1_thresh    = args.rel_l1_thresh

    print(
        f"[teacache-flux] rel_l1_thresh={args.rel_l1_thresh}  "
        f"steps={args.num_inference_steps}  seeds={seeds}",
        flush=True,
    )

    per_seed_quality: dict[str, dict] = {}

    for seed in seeds:
        # TeaCache 状态必须在每次生成前重置
        reset_teacache_state(transformer)

        generator = torch.Generator("cuda").manual_seed(seed)
        with get_timer("inference"):
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
        print(f"[teacache-flux] Saved {output_path}", flush=True)

        if not args.disable_quality_eval:
            with get_timer("quality_eval"):
                quality_result = evaluate_quality_with_origin(
                    image_path=output_path,
                    tag=TAG,
                    save_metrics=False,
                    baseline_path=None,
                    output_dir=output_dir,
                )
            per_seed_quality[f"seed{seed}"] = (
                quality_result["metrics"] if quality_result is not None else {}
            )

    quality_metrics = None
    if not args.disable_quality_eval:
        valid_metrics = [m for m in per_seed_quality.values() if m]
        if valid_metrics:
            metric_names = ["psnr", "ssim", "lpips", "fid", "fvd"]
            mean_metrics = {}
            for name in metric_names:
                values = [float(m[name]) for m in valid_metrics if m.get(name) is not None]
                mean_metrics[name] = mean(values) if values else None
            quality_metrics = {"per_seed": per_seed_quality, "mean": mean_metrics}
        else:
            quality_metrics = {"per_seed": per_seed_quality, "mean": None}

    print_time_statistics()

    params = {
        "model":               "flux",
        "method":              TAG,
        "model_id":            model_id,
        "model_path":          args.model_path,
        "prompt":              args.prompt,
        "height":              args.height,
        "width":               args.width,
        "num_images":          len(seeds),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale":      args.guidance_scale,
        "seeds":               seeds,
        "rel_l1_thresh":       args.rel_l1_thresh,
        "quality_eval":        not args.disable_quality_eval,
    }
    params_path = save_params_and_metrics(
        output_dir,
        TAG,
        params,
        get_time_statistics_dict(),
        quality_metrics=quality_metrics,
    )
    print(f"[teacache-flux] Params & metrics saved to {params_path}", flush=True)


if __name__ == "__main__":
    main()
