# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
批量 Wan T2V ToCa（tocache）：按行读取 prompt，输出目录为
results/wan/{model_id}/tocache_{a{attn}_c{cross}_f{ffn}}/{prompt_id}/。
其中 attn / cross-attn / ffn 的强制全算周期从
`wan/jano_baselines/tocache/force_scheduler.py` 的 `cache_dic['cal_threshold']`
读取（与单次推理运行时一致）；不在命令行上做参数网格遍历。

可与 results/wan/{model_id}/ori/{prompt_id}/ 下 ori 视频比对质量（需传 --ori_baseline_root）。

用法（Cubic 根目录）:
  python wan_exp/tocache_batch_generate.py

  python wan_exp/tocache_batch_generate.py \\
    --prompts_file data/remaining_prompts.txt --sample_steps 50 \\
    --ori_baseline_root results/wan/1.3B/ori

  # 仅覆盖目录后缀（仍使用 force_scheduler 中的实际调度）
  python wan_exp/tocache_batch_generate.py --tocache_run_subdir custom_suffix

集群:
  bash wan_exp/batch_wan_tocache.sh -- --prompts_file data/remaining_prompts.txt \\
    --sample_steps 50 --ori_baseline_root results/wan/1.3B/ori

单机后台:
  bash wan_exp/run_tocache_batch_nohup.sh
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


def read_cal_threshold_from_force_scheduler() -> dict[str, int]:
    """与运行时一致：执行 force_scheduler(cache_dic, current) 得到 cal_threshold。

    通过文件路径加载模块，避免 `import wan...` 拉取 WAN 大包（及其 torch 依赖），
    便于在未完整安装推理环境时仍可解析后缀。
    """
    import importlib.util

    _wan_exp = os.path.join(os.path.dirname(__file__))
    fp = os.path.join(
        _wan_exp, "wan", "jano_baselines", "tocache", "force_scheduler.py"
    )
    if not os.path.isfile(fp):
        raise FileNotFoundError(f"未找到 force_scheduler: {fp}")
    spec = importlib.util.spec_from_file_location(
        "tocache_force_scheduler_dyn", fp
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {fp}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    force_scheduler = getattr(mod, "force_scheduler", None)
    if force_scheduler is None:
        raise RuntimeError(f"{fp} 中缺少 force_scheduler 函数")

    cache_dic: dict[str, object] = {}
    force_scheduler(cache_dic, {})
    raw = cache_dic.get("cal_threshold")
    if not isinstance(raw, dict):
        raise RuntimeError(
            "force_scheduler 未设置有效的 cache_dic['cal_threshold'] dict"
        )
    out: dict[str, int] = {}
    for k, v in raw.items():
        if not isinstance(v, int):
            raise RuntimeError(f"cal_threshold[{k!r}] 应为 int，得到 {type(v)}")
        out[str(k)] = v
    return out


def _tocache_suffix_from_scheduler(cal_threshold: dict[str, int]) -> str:
    """cal_threshold 键名与 force_scheduler.py 一致：attn / cross-attn / ffn。"""
    try:
        a = cal_threshold["attn"]
        c = cal_threshold["cross-attn"]
        f = cal_threshold["ffn"]
    except KeyError as e:
        missing = {"attn", "cross-attn", "ffn"} - cal_threshold.keys()
        raise RuntimeError(
            "cal_threshold 缺少键 attn / cross-attn / ffn;"
            f" 当前键: {sorted(cal_threshold)!r}; 缺失: {sorted(missing)!r}"
        ) from e
    return f"a{a}_c{c}_f{f}"


def _parse_batch_cli() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description="Batch Wan ToCa via tocache_generate.py.")
    p.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help=f"每行一条 prompt；默认 {_DEFAULT_PROMPTS}",
    )
    p.add_argument(
        "--ori_baseline_root",
        type=str,
        default=None,
        help="ori 结果根目录；默认 results/wan/{model_id}/ori",
    )
    p.add_argument(
        "--tocache_run_subdir",
        type=str,
        default=None,
        help="覆盖默认的 a{attn}_c{cross}_f{ffn} 后缀（目录仍为 tocache_<后缀>）。",
    )
    batch_args, rest = p.parse_known_args(sys.argv[1:])
    while rest and rest[0] == "--":
        rest = rest[1:]

    tc_only: list[str] = []
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
        if item == "--ori_baseline_root" and i + 1 < len(rest):
            batch_args.ori_baseline_root = rest[i + 1]
            i += 2
            continue
        if item.startswith("--ori_baseline_root="):
            batch_args.ori_baseline_root = item.split("=", 1)[1]
            i += 1
            continue
        if item == "--tocache_run_subdir" and i + 1 < len(rest):
            batch_args.tocache_run_subdir = rest[i + 1]
            i += 2
            continue
        if item.startswith("--tocache_run_subdir="):
            batch_args.tocache_run_subdir = item.split("=", 1)[1]
            i += 1
            continue
        tc_only.append(item)
        i += 1

    if batch_args.prompts_file is None:
        batch_args.prompts_file = _DEFAULT_PROMPTS
    return batch_args, tc_only


def main() -> None:
    os.chdir(_CUBIC_ROOT)
    batch_args, tc_argv_rest = _parse_batch_cli()
    prompts_path = os.path.abspath(batch_args.prompts_file)
    if not os.path.isfile(prompts_path):
        raise FileNotFoundError(f"prompt 文件不存在: {prompts_path}")

    prompts = _load_prompts(prompts_path)
    if not prompts:
        raise RuntimeError(f"未从 {prompts_path} 读到任何有效 prompt 行")

    cal_threshold = read_cal_threshold_from_force_scheduler()
    suffix_default = _tocache_suffix_from_scheduler(cal_threshold)
    suffix = batch_args.tocache_run_subdir or suffix_default

    import tocache_generate as tc
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
                f"\n========== [{idx}/{len(prompts)}] tocache_{suffix} "
                f"(cal_threshold={cal_threshold}) ==========\n{prompt}\n",
                flush=True,
            )

        tc.PROMPT = prompt
        tc.OUTPUT_DIR = (
            f"./results/wan/{tc.model_id}/tocache_{suffix}/{tc.get_prompt_id(prompt)}"
        )

        sys.argv = [sys.argv[0]] + [
            "--ori_baseline_root",
            ori_root_pass,
        ] + tc_argv_rest

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
                "method": "tocache",
                "tocache_run_subdir": suffix,
                "tocache_cal_threshold": dict(cal_threshold),
                "tocache_force_scheduler_source": (
                    "wan/jano_baselines/tocache/force_scheduler.py"
                ),
                "model_id": tc.model_id,
                "model_path": tc.MODEL_PATH,
                "prompt": tc.PROMPT,
                "task": args.task,
                "size": args.size,
                "base_seed": args.base_seed,
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
