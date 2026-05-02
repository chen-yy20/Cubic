# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
批量 Wan T2V PAB：按行读取 prompt，输出目录为 results/wan/{model_id}/pab_s{self}_c{cross}/{prompt_id}/，
并与 results/wan/{model_id}/ori/{prompt_id}/ 下的 ori 视频做质量对比。

self / cross 对应 pab_generate 中 --pab_self_range、--pab_cross_range（self-attn / cross-attn 广播间隔）。

用法（Cubic 根目录）:
  python wan_exp/pab_batch_generate.py --pab_self_range 2 --pab_cross_range 5

  bash wan_exp/batch_wan_pab.sh -- --pab_self_range 4 --pab_cross_range 5 \\
    --prompts_file data/remaining_prompts.txt --sample_steps 50

多组间隔后台提交见 wan_exp/run_pab_interval_sweep_nohup.sh（默认 (2,5)、(4,5)、(6,6)）。
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


def _pab_interval_subdir(self_range: int, cross_range: int) -> str:
    """如 self=2 cross=5 -> s2_c5，目录为 pab_s2_c5。"""
    return f"s{self_range}_c{cross_range}"


def _parse_batch_cli() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description="Batch Wan PAB (pab_generate.py).")
    p.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help=f"每行一条 prompt；默认 {_DEFAULT_PROMPTS}",
    )
    p.add_argument(
        "--pab_self_range",
        type=int,
        default=2,
        help="Self-attention 间隔（与 pab_generate 一致）。",
    )
    p.add_argument(
        "--pab_cross_range",
        type=int,
        default=5,
        help="Cross-attention 间隔（与 pab_generate 一致）。",
    )
    p.add_argument(
        "--ori_baseline_root",
        type=str,
        default=None,
        help="ori 结果根目录；默认 results/wan/{model_id}/ori",
    )
    p.add_argument(
        "--pab_run_subdir",
        type=str,
        default=None,
        help="覆盖默认 pab_s{self}_c{cross} 中的后缀（如自定义实验名）。",
    )
    batch_args, rest = p.parse_known_args(sys.argv[1:])
    while rest and rest[0] == "--":
        rest = rest[1:]

    pab_only: list[str] = []
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
        if item == "--pab_self_range" and i + 1 < len(rest):
            batch_args.pab_self_range = int(rest[i + 1])
            i += 2
            continue
        if item.startswith("--pab_self_range="):
            batch_args.pab_self_range = int(item.split("=", 1)[1])
            i += 1
            continue
        if item == "--pab_cross_range" and i + 1 < len(rest):
            batch_args.pab_cross_range = int(rest[i + 1])
            i += 2
            continue
        if item.startswith("--pab_cross_range="):
            batch_args.pab_cross_range = int(item.split("=", 1)[1])
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
        if item == "--pab_run_subdir" and i + 1 < len(rest):
            batch_args.pab_run_subdir = rest[i + 1]
            i += 2
            continue
        if item.startswith("--pab_run_subdir="):
            batch_args.pab_run_subdir = item.split("=", 1)[1]
            i += 1
            continue
        pab_only.append(item)
        i += 1

    if batch_args.prompts_file is None:
        batch_args.prompts_file = _DEFAULT_PROMPTS
    return batch_args, pab_only


def main() -> None:
    os.chdir(_CUBIC_ROOT)
    batch_args, pab_argv_rest = _parse_batch_cli()
    prompts_path = os.path.abspath(batch_args.prompts_file)
    if not os.path.isfile(prompts_path):
        raise FileNotFoundError(f"prompt 文件不存在: {prompts_path}")

    prompts = _load_prompts(prompts_path)
    if not prompts:
        raise RuntimeError(f"未从 {prompts_path} 读到任何有效 prompt 行")

    suffix = batch_args.pab_run_subdir or _pab_interval_subdir(
        batch_args.pab_self_range, batch_args.pab_cross_range
    )

    import pab_generate as pab
    from utils.results import save_params_and_metrics
    from utils.timer import (
        get_time_statistics_dict,
        init_timer,
        print_time_statistics,
        save_time_stats_to_file,
    )

    if batch_args.ori_baseline_root is None:
        batch_args.ori_baseline_root = os.path.join(
            "results", "wan", pab.model_id, "ori"
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
                f"\n========== [{idx}/{len(prompts)}] pab_{suffix} "
                f"(self={batch_args.pab_self_range}, cross={batch_args.pab_cross_range}) ==========\n"
                f"{prompt}\n",
                flush=True,
            )

        pab.PROMPT = prompt
        pab.OUTPUT_DIR = (
            f"./results/wan/{pab.model_id}/pab_{suffix}/{pab.get_prompt_id(prompt)}"
        )

        sys.argv = [sys.argv[0]] + [
            "--pab_self_range",
            str(batch_args.pab_self_range),
            "--pab_cross_range",
            str(batch_args.pab_cross_range),
            "--ori_baseline_root",
            ori_root_pass,
        ] + pab_argv_rest

        args = pab._parse_args()
        quality_metrics = pab.generate(args)
        print_time_statistics(timer_names=pab.REQUIRED_TIMER_NAMES)

        if rank == 0:
            out_dir = getattr(args, "output_dir", None) or pab.OUTPUT_DIR
            save_time_stats_to_file(
                os.path.join(out_dir, "timing_stats.txt"),
                timer_names=pab.REQUIRED_TIMER_NAMES,
            )
            params = {
                "model": "wan",
                "method": "pab",
                "pab_run_subdir": suffix,
                "model_id": pab.model_id,
                "model_path": pab.MODEL_PATH,
                "prompt": pab.PROMPT,
                "task": args.task,
                "size": args.size,
                "base_seed": args.base_seed,
                "pab_self_range": args.pab_self_range,
                "pab_cross_range": args.pab_cross_range,
                "pab_warmup": args.pab_warmup,
                "pab_cooldown": args.pab_cooldown,
                "pab_layer_interval": args.pab_layer_interval,
                "ori_baseline_root": ori_root_pass,
            }
            out = save_params_and_metrics(
                out_dir,
                pab.TAG,
                params,
                get_time_statistics_dict(timer_names=pab.REQUIRED_TIMER_NAMES),
                quality_metrics=quality_metrics,
            )
            print(f"Params & metrics saved to {out}", flush=True)

        init_timer()


if __name__ == "__main__":
    main()
