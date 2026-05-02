# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
批量 Wan T2V ori 实验：按行读取 prompt 文件，复用 `ori_generate.py` 中的流程。

不修改 `ori_generate.py`：在每次迭代前更新该模块内的 PROMPT / OUTPUT_DIR，再调用其 `_parse_args` 与 `generate`。

用法（在 Cubic 仓库根目录，且已设置 PYTHONPATH 包含仓库根，与 `ori_generate.py` 一致）:
  python wan_exp/ori_batch_generate.py
  python wan_exp/ori_batch_generate.py --prompts_file data/remaining_prompts.txt --sample_steps 50

双卡与 `ori_generate.py` 相同，例如:
  torchrun --standalone --nproc_per_node=2 wan_exp/ori_batch_generate.py

集群提交（与 `launch.py` 一致）:
  python launch.py --model wan --method ori_batch --partition debug --gpus 2 -- --prompts_file data/remaining_prompts.txt
  bash wan_exp/batch_wan_ori.sh -- --sample_steps 50
"""
from __future__ import annotations

import argparse
import os
import sys

# 本文件位于 wan_exp/，仓库根为其上级目录
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


def _parse_batch_cli() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Batch Wan ori generation using ori_generate.py (unmodified)."
    )
    p.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help=f"每行一条 prompt；默认 {_DEFAULT_PROMPTS}",
    )
    batch_args, rest = p.parse_known_args(sys.argv[1:])

    # launch.py 常把参数写成 `-- --prompts_file ...`：第一个 `--` 会让本 parser
    # 无法吞掉后面的 `--prompts_file`，整段会落在 rest 里；若原样交给
    # ori_generate._parse_args() 会报 unrecognized arguments。
    while rest and rest[0] == "--":
        rest = rest[1:]

    ori_only: list[str] = []
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
        ori_only.append(item)
        i += 1

    if batch_args.prompts_file is None:
        batch_args.prompts_file = _DEFAULT_PROMPTS
    return batch_args, ori_only


def main() -> None:
    batch_args, ori_argv_rest = _parse_batch_cli()
    prompts_path = os.path.abspath(batch_args.prompts_file)
    if not os.path.isfile(prompts_path):
        raise FileNotFoundError(f"prompt 文件不存在: {prompts_path}")

    prompts = _load_prompts(prompts_path)
    if not prompts:
        raise RuntimeError(f"未从 {prompts_path} 读到任何有效 prompt 行")

    # 交给 ori_generate 的参数（与原脚本一致）
    sys.argv = [sys.argv[0]] + ori_argv_rest

    import ori_generate as ori
    from utils.results import save_params_and_metrics
    from utils.timer import get_time_statistics_dict, init_timer, print_time_statistics

    rank = int(os.getenv("RANK", 0))

    for idx, prompt in enumerate(prompts, start=1):
        if rank == 0:
            print(f"\n========== [{idx}/{len(prompts)}] ==========\n{prompt}\n", flush=True)

        ori.PROMPT = prompt
        ori.OUTPUT_DIR = (
            f"./results/wan/{ori.model_id}/ori/{ori.get_prompt_id(prompt)}"
        )

        args = ori._parse_args()
        ori.generate(args)
        print_time_statistics()

        if rank == 0:
            params = {
                "model": "wan",
                "method": "ori",
                "model_id": ori.model_id,
                "model_path": ori.MODEL_PATH,
                "prompt": ori.PROMPT,
                "task": args.task,
                "size": args.size,
                "base_seed": args.base_seed,
            }
            out = save_params_and_metrics(
                ori.OUTPUT_DIR,
                ori.TAG,
                params,
                get_time_statistics_dict(),
                quality_metrics=None,
            )
            print(f"Params & metrics saved to {out}", flush=True)

        init_timer()


if __name__ == "__main__":
    main()
