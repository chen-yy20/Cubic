# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
批量 Wan T2V TeaCache：按行读取 prompt，输出目录为 results/wan/{model_id}/teacache_t0pXX/{prompt_id}/，
并与 results/wan/{model_id}/ori/{prompt_id}/ 下的 ori 视频做质量对比。

用法（在 Cubic 仓库根目录）:
  python wan_exp/teacache_batch_generate.py --teacache_thresh 0.05
  python wan_exp/teacache_batch_generate.py \\
    --prompts_file data/remaining_prompts.txt --teacache_thresh 0.07 --sample_steps 50

  # 自定义相对 ori 根目录（默认 results/wan/{model_id}/ori）:
  python wan_exp/teacache_batch_generate.py --teacache_thresh 0.08 \\
    --ori_baseline_root results/wan/1.3B/ori

双卡:
  torchrun --standalone --nproc_per_node=2 wan_exp/teacache_batch_generate.py --teacache_thresh 0.05

集群:
  bash wan_exp/batch_wan_teacache.sh -- --teacache_thresh 0.05 --prompts_file data/remaining_prompts.txt
"""
from __future__ import annotations

import argparse
import os
import sys

_CUBIC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_DEFAULT_PROMPTS = os.path.join(_CUBIC_ROOT, "data", "remaining_prompts.txt")


def _load_prompts(path: str) -> list[str]:
    prompts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip().rstrip("\r")
            if not line or line.startswith("#"):
                continue
            prompts.append(line)
    return prompts


def _thresh_subdir(th: float) -> str:
    """0.05 -> t0p05，与目录名 teacache_t0p05 对应。"""
    return f"t{th:.2f}".replace(".", "p")


def _parse_batch_cli() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Batch Wan TeaCache generation (teacache_generate.py)."
    )
    p.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help=f"每行一条 prompt；默认 {_DEFAULT_PROMPTS}",
    )
    p.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.07,
        help="TeaCache 阈值；输出目录名会包含该值（teacache_t0pXX）。",
    )
    p.add_argument(
        "--ori_baseline_root",
        type=str,
        default=None,
        help="ori 结果根目录（内含各 prompt_id 子目录）；默认 results/wan/{model_id}/ori",
    )
    p.add_argument(
        "--teacache_subdir",
        type=str,
        default=None,
        help="覆盖默认的 teacache_t{thresh} 中的后缀（默认由 --teacache_thresh 生成）。",
    )
    batch_args, rest = p.parse_known_args(sys.argv[1:])
    while rest and rest[0] == "--":
        rest = rest[1:]

    te_only: list[str] = []
    i = 0
    while i < len(rest):
        item = rest[i]
        if item == "--prompts_file" and i + 1 < len(rest):
            batch_args.prompts_file = rest[i + 1]
            i += 2
            continue
        if item.startswith("--prompts_file="):
            batch_args.prompts_file = item.split("=", 1)[1]
            i += 1
            continue
        if item == "--teacache_thresh" and i + 1 < len(rest):
            batch_args.teacache_thresh = float(rest[i + 1])
            i += 2
            continue
        if item.startswith("--teacache_thresh="):
            batch_args.teacache_thresh = float(item.split("=", 1)[1])
            i += 1
            continue
        if item == "--ori_baseline_root" and i + 1 < len(rest):
            batch_args.ori_baseline_root = rest[i + 1]
            i += 2
            continue
        if item.startswith("--ori_baseline_root="):
            batch_args.ori_baseline_root = item.split("=", 1)[1]
            i += 1
            continue
        if item == "--teacache_subdir" and i + 1 < len(rest):
            batch_args.teacache_subdir = rest[i + 1]
            i += 2
            continue
        if item.startswith("--teacache_subdir="):
            batch_args.teacache_subdir = item.split("=", 1)[1]
            i += 1
            continue
        te_only.append(item)
        i += 1

    if batch_args.prompts_file is None:
        batch_args.prompts_file = _DEFAULT_PROMPTS
    return batch_args, te_only


def main() -> None:
    os.chdir(_CUBIC_ROOT)
    batch_args, te_argv_rest = _parse_batch_cli()
    prompts_path = os.path.abspath(batch_args.prompts_file)
    if not os.path.isfile(prompts_path):
        raise FileNotFoundError(f"prompt 文件不存在: {prompts_path}")

    prompts = _load_prompts(prompts_path)
    if not prompts:
        raise RuntimeError(f"未从 {prompts_path} 读到任何有效 prompt 行")

    suffix = batch_args.teacache_subdir or _thresh_subdir(batch_args.teacache_thresh)

    import teacache_generate as tc
    from utils.results import save_params_and_metrics
    from utils.timer import (
        get_time_statistics_dict,
        init_timer,
        print_time_statistics,
        save_time_stats_to_file,
    )

    if batch_args.ori_baseline_root is None:
        batch_args.ori_baseline_root = os.path.join(
            "results", "wan", tc.model_id, "ori"
        )

    ori_root_pass = os.path.abspath(
        os.path.join(_CUBIC_ROOT, batch_args.ori_baseline_root)
        if not os.path.isabs(batch_args.ori_baseline_root)
        else batch_args.ori_baseline_root
    )

    rank = int(os.getenv("RANK", 0))

    for idx, prompt in enumerate(prompts, start=1):
        if rank == 0:
            print(
                f"\n========== [{idx}/{len(prompts)}] teacache_t{suffix} ==========\n{prompt}\n",
                flush=True,
            )

        tc.PROMPT = prompt
        tc.OUTPUT_DIR = (
            f"./results/wan/{tc.model_id}/teacache_{suffix}/{tc.get_prompt_id(prompt)}"
        )

        sys.argv = [sys.argv[0]] + [
            "--teacache_thresh",
            str(batch_args.teacache_thresh),
            "--ori_baseline_root",
            ori_root_pass,
        ] + te_argv_rest

        args = tc._parse_args()
        quality_metrics = tc.generate(args)
        print_time_statistics(timer_names=tc.REQUIRED_TIMER_NAMES)

        if rank == 0:
            out_dir = getattr(args, "output_dir", None) or tc.OUTPUT_DIR
            save_time_stats_to_file(
                os.path.join(out_dir, "timing_stats.txt"),
                timer_names=tc.REQUIRED_TIMER_NAMES,
            )
            params = {
                "model": "wan",
                "method": "teacache",
                "teacache_subdir": suffix,
                "model_id": tc.model_id,
                "model_path": tc.MODEL_PATH,
                "prompt": tc.PROMPT,
                "task": args.task,
                "size": args.size,
                "base_seed": args.base_seed,
                "teacache_thresh": args.teacache_thresh,
                "use_ret_steps": args.use_ret_steps,
                "ori_baseline_root": ori_root_pass,
            }
            out = save_params_and_metrics(
                out_dir,
                tc.TAG,
                params,
                get_time_statistics_dict(timer_names=tc.REQUIRED_TIMER_NAMES),
                quality_metrics=quality_metrics,
            )
            print(f"Params & metrics saved to {out}", flush=True)

        init_timer()


if __name__ == "__main__":
    main()
