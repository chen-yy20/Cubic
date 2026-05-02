#!/usr/bin/env python3
"""
Cubic 项目 VBench 测评入口：
- 从完整 VBench prompt JSON 中：随机抽 n 条，或按文本文件（每行一条 prompt_en）筛选；
- 对「全维度」依次调用官方 VBench.evaluate。

前置：在有网机器运行 download_weights.py，并设置 VBENCH_CACHE_DIR。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_vbench_on_path(vbench_root: Path) -> None:
    root = vbench_root.resolve()
    if not (root / "vbench" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"未找到 VBench 包: 期望 {root / 'vbench' / '__init__.py'}。"
            "请通过 --vbench-root 指定 VBench-master 根目录。"
        )
    parent = str(root)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def sample_full_info(
    full_info: list,
    n: int,
    seed: int,
) -> list:
    if n <= 0 or n >= len(full_info):
        return list(full_info)
    rng = random.Random(seed)
    return rng.sample(full_info, n)


def read_prompt_lines(path: Path) -> list[str]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    return lines


def subset_full_info_from_prompt_lines(
    full_info: list,
    prompt_lines: list[str],
) -> tuple[list, list[str]]:
    """
    按 prompt_en 与 txt 每行**完全一致**匹配（区分大小写与标点）。
    返回 (匹配的条目列表, 在 JSON 中找不到的 txt 行)。
    """
    index = {}
    for entry in full_info:
        key = entry.get("prompt_en")
        if key not in index:
            index[key] = entry
    selected: list = []
    missing: list[str] = []
    seen: set[str] = set()
    for line in prompt_lines:
        if line in seen:
            continue
        seen.add(line)
        if line in index:
            selected.append(index[line])
        else:
            missing.append(line)
    return selected, missing


def full_dimension_list() -> list[str]:
    return [
        "subject_consistency",
        "background_consistency",
        "aesthetic_quality",
        "imaging_quality",
        "object_class",
        "multiple_objects",
        "color",
        "spatial_relationship",
        "scene",
        "temporal_style",
        "overall_consistency",
        "human_action",
        "temporal_flickering",
        "motion_smoothness",
        "dynamic_degree",
        "appearance_style",
    ]


def main() -> None:
    default_vbench = _repo_root() / "VBench-master"
    default_full_json = default_vbench / "vbench" / "VBench_full_info.json"

    parser = argparse.ArgumentParser(
        description="Cubic VBench：子集 prompt + 全维度测评（离线权重）"
    )
    parser.add_argument(
        "--vbench-root",
        type=str,
        default=str(default_vbench),
        help="VBench-master 根目录（内含 vbench 包）",
    )
    parser.add_argument(
        "--full-json",
        type=str,
        default=str(default_full_json),
        help="完整 prompt 定义 JSON（官方 VBench_full_info.json，或自拷贝为 vbench_full.json）",
    )
    parser.add_argument(
        "--subset-json",
        type=str,
        default="",
        help="若指定则直接使用该子集 JSON；忽略 --prompts-file / --n / --seed",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="",
        help="文本文件，每行一条 prompt_en，与 VBench_full_info.json 中字段完全一致；优先于随机 --n",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="未指定 --prompts-file / --subset-json 时：从 full-json 随机抽 n 条；<=0 表示全部",
    )
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    parser.add_argument(
        "--videos-path",
        type=str,
        default="",
        help="生成视频目录（--dry-run 时可省略）；命名需与 VBench 一致：{prompt}-0.mp4 ...",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
        help="结果输出目录；默认 Cubic/evaluation_results/vbench_subset/",
    )
    parser.add_argument(
        "--dimension",
        nargs="*",
        default=None,
        help="不传则测评全部标准维度；传入则只跑指定维度",
    )
    parser.add_argument(
        "--name-prefix",
        type=str,
        default="cubic_vbench",
        help="输出文件名前缀（会附加时间戳）",
    )
    parser.add_argument(
        "--local",
        dest="local",
        action="store_true",
        default=True,
        help="从本地缓存加载 CLIP/DINO 等（离线节点必须；默认开启）",
    )
    parser.add_argument(
        "--no-local",
        dest="local",
        action="store_false",
        help="关闭 local 模式（可能触发在线拉取，不推荐离线环境）",
    )
    parser.add_argument(
        "--read-frame",
        action="store_true",
        help="与官方 evaluate 一致：从帧目录读图而非视频",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅写出子集 JSON，不运行测评",
    )
    parser.add_argument(
        "--video-extension",
        type=str,
        default=".mp4",
        help="视频后缀（默认 .mp4）。目录中有 metadata.jsonl 等杂文件时须固定此项，避免官方误用首个文件的扩展名。",
    )
    args = parser.parse_args()

    out = (
        Path(args.output_path).resolve()
        if args.output_path
        else _repo_root() / "evaluation_results" / "vbench_subset"
    )
    out.mkdir(parents=True, exist_ok=True)

    full_json_path = Path(args.full_json).resolve()
    if not full_json_path.is_file():
        raise FileNotFoundError(f"找不到 full JSON: {full_json_path}")

    with open(full_json_path, "r", encoding="utf-8") as f:
        full_info = json.load(f)

    run_tag_seed = args.seed

    if args.subset_json:
        subset_path = Path(args.subset_json).resolve()
        if not subset_path.is_file():
            raise FileNotFoundError(f"找不到 subset JSON: {subset_path}")
    elif args.prompts_file:
        pf = Path(args.prompts_file).resolve()
        if not pf.is_file():
            raise FileNotFoundError(f"找不到 prompts 文件: {pf}")
        lines = read_prompt_lines(pf)
        subset, missing_lines = subset_full_info_from_prompt_lines(full_info, lines)
        if missing_lines:
            print(
                "警告：下列行在 full-json 中找不到完全一致的 prompt_en "
                f"（共 {len(missing_lines)} 条），已跳过："
            )
            for m in missing_lines[:20]:
                print(f"  - {m!r}")
            if len(missing_lines) > 20:
                print(f"  ... 另有 {len(missing_lines) - 20} 条")
        if not subset:
            raise SystemExit(
                "没有任何条目匹配成功。请检查 txt 是否与 VBench_full_info.json 里 "
                "`prompt_en` 字符串完全一致（含标点、空格）。"
            )
        stem = pf.stem
        subset_path = out / f"subset_prompts_{stem}_n{len(subset)}_full_info.json"
        with open(subset_path, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        print(f"已从 prompts 文件写入子集 ({len(subset)} 条): {subset_path}")
        run_tag_seed = -1
    else:
        subset = sample_full_info(full_info, args.n, args.seed)
        n_tag = len(full_info) if args.n <= 0 or args.n >= len(full_info) else args.n
        subset_path = out / f"subset_n{n_tag}_seed{args.seed}_full_info.json"
        with open(subset_path, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        print(f"已写入子集 ({len(subset)} 条): {subset_path}")

    if args.dry_run:
        print("dry-run：跳过测评。")
        return
    if not args.videos_path:
        raise SystemExit("请提供 --videos-path，或使用 --dry-run 仅生成子集 JSON。")

    _ensure_vbench_on_path(Path(args.vbench_root))

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    import torch
    from vbench.distributed import dist_init, print0

    from cubic_vbench import CubicVBench

    dist_init()
    device = torch.device("cuda")
    dimension_list = args.dimension if args.dimension else full_dimension_list()

    with open(subset_path, "r", encoding="utf-8") as f:
        subset_entries = json.load(f)
    n_entries = len(subset_entries)

    ts = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    seed_part = f"seed{args.seed}" if run_tag_seed >= 0 else "promptsfile"
    run_name = f"{args.name_prefix}_{ts}_n{n_entries}_{seed_part}"

    print0(f"VBENCH_CACHE_DIR={os.environ.get('VBENCH_CACHE_DIR', '(默认 ~/.cache/vbench)')}")
    print0(f"subset json: {subset_path}")
    print0(f"videos_path: {args.videos_path}")
    print0(f"video_extension: {args.video_extension}")
    print0(f"dimensions ({len(dimension_list)}): {dimension_list}")

    my_vbench = CubicVBench(device, str(subset_path), str(out))
    my_vbench.evaluate(
        videos_path=args.videos_path,
        name=run_name,
        prompt_list=[],
        dimension_list=dimension_list,
        local=args.local,
        read_frame=args.read_frame,
        mode="vbench_standard",
        video_extension=args.video_extension,
    )
    print0("完成。")


if __name__ == "__main__":
    main()
