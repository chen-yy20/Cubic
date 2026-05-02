#!/usr/bin/env python3
"""
使用 VBench 对自定义视频集合进行自定义模式评估，并生成包含平均得分与逐视频得分的 JSON。

主要流程：
1. 读取 prompt.txt（支持多种分隔符），获得需要评估的视频及其文本描述。
2. 依据 prompt 过滤目标文件夹内的视频，仅保留匹配项。
3. 调用 VBench 的自定义评估模式，对所有支持 custom_input 的维度逐一打分。
4. 重新整理官方输出，生成包含每个维度平均得分和逐视频得分的汇总 JSON。
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import tempfile
import shutil
import os

from vbench import VBench
from vbench.utils import get_prompt_from_filename, load_json, save_json


# 默认仅评估 subject_consistency 与 imaging_quality，可通过 --dimensions 覆盖
DEFAULT_DIMENSIONS: Tuple[str, ...] = (
    "subject_consistency",
    "imaging_quality",
)


def parse_prompt_file(prompt_path: Path) -> List[Tuple[str, str]]:
    """解析 prompt.txt，返回 (键, prompt) 列表。

    支持分隔符优先级：TAB / 管道符 / 逗号。若不存在分隔符，则认为该行即为 prompt。
    键可以是文件名、去后缀的文件名，或 prompt 本身。
    """
    pairs: List[Tuple[str, str]] = []
    with prompt_path.open("r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for sep in ("\t", "|", ","):
                if sep in line:
                    key, prompt = line.split(sep, 1)
                    break
            else:
                key = line
                prompt = line
            key = key.strip()
            prompt = prompt.strip()
            if not key or not prompt:
                raise ValueError(f"第 {line_idx} 行格式非法：{line!r}")
            pairs.append((key, prompt))
    if not pairs:
        raise ValueError(f"在 {prompt_path} 中未找到有效的 prompt 行")
    return pairs


def collect_videos(video_dir: Path) -> Dict[str, Path]:
    """收集目录下所有可评估的视频/动图文件。"""
    allowed_suffix = {".mp4", ".gif", ".png"}
    videos: Dict[str, Path] = {}
    for item in video_dir.iterdir():
        if item.is_file() and item.suffix.lower() in allowed_suffix:
            videos[item.name] = item.resolve()
    if not videos:
        raise FileNotFoundError(f"{video_dir} 中未找到可评估的视频文件（mp4/gif/png）")
    return videos


def match_prompts_to_videos(
    video_dir: Path, raw_entries: Iterable[Tuple[str, str]]
) -> Dict[str, str]:
    """依据 prompt 文件条目筛选并匹配视频。

    返回 {文件名: prompt} 字典，同时允许：
    - 使用完整文件名匹配
    - 使用去后缀文件名匹配
    - 使用通过 VBench 的 get_prompt_from_filename 还原后的文本匹配
    """
    videos = collect_videos(video_dir)
    filename_map = {Path(name).stem: name for name in videos}
    matched: Dict[str, str] = {}
    unmatched: List[str] = []

    for key, prompt in raw_entries:
        candidates: List[str] = []

        if key in videos:
            candidates = [key]
        elif key in filename_map:
            candidates = [filename_map[key]]
        else:
            # 尝试通过 prompt 还原匹配
            restored_matches = [
                name for name in videos if get_prompt_from_filename(name).strip() == key
            ]
            if restored_matches:
                candidates = restored_matches
            else:
                prompt_matches = [
                    name
                    for name in videos
                    if get_prompt_from_filename(name).strip() == prompt
                ]
                if prompt_matches:
                    candidates = prompt_matches

        if not candidates:
            unmatched.append(key)
            continue

        for name in candidates:
            matched[name] = prompt

    if not matched:
        raise ValueError(
            "未能根据 prompt.txt 匹配到任何视频，请检查文件名或分隔符格式是否正确。"
        )

    if unmatched:
        print(
            f"[警告] 以下 {len(unmatched)} 条记录未匹配到视频：{unmatched}",
            file=sys.stderr,
        )

    return matched


def create_subset_directory(
    source_dir: Path, prompt_map: Dict[str, str], work_dir: Path
) -> Tuple[Path, Dict[str, str]]:
    """将命中的视频软链接到临时子目录，返回子目录路径与新 prompt map。

    这样可以保证 VBench 仅看到需要评估的视频，避免 prompt 数量与视频数不一致。
    """
    temp_dir = Path(
        tempfile.mkdtemp(prefix="vbench_subset_", dir=str(work_dir.resolve()))
    )
    subset_map: Dict[str, str] = {}
    for filename, prompt in prompt_map.items():
        src = source_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"找不到视频文件：{src}")
        dst = temp_dir / filename
        if not dst.exists():
            try:
                os.symlink(src, dst)
            except (AttributeError, NotImplementedError, OSError):
                shutil.copy2(src, dst)
        subset_map[dst.name] = prompt
    return temp_dir, subset_map


def run_vbench_evaluation(
    videos_path: Path,
    prompt_map: Dict[str, str],
    output_dir: Path,
    result_name: str,
    dimensions: Tuple[str, ...],
    full_json_path: Path,
    load_ckpt_from_local: bool = False,
    read_frame: bool = False,
) -> Path:
    """调用 VBench 进行自定义模式评估，返回官方结果 JSON 路径。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print(
            "[提示] 当前未检测到可用 CUDA，将使用 CPU 进行推理，速度可能较慢，"
            "若部分模型仅支持 GPU 可能会触发错误。",
            file=sys.stderr,
        )

    subset_dir, subset_prompt_map = create_subset_directory(videos_path, prompt_map, output_dir)

    try:
        vbench = VBench(device, str(full_json_path), str(output_dir))
        debug_files = list(subset_dir.iterdir())
        print(f"[调试] 子目录: {subset_dir}, 文件数: {len(debug_files)}, prompt 数: {len(subset_prompt_map)}")
        vbench.evaluate(
            videos_path=str(subset_dir),
            name=result_name,
            prompt_list=subset_prompt_map,
            dimension_list=list(dimensions),
            mode="custom_input",
            local=load_ckpt_from_local,
            read_frame=read_frame,
        )
    finally:
        shutil.rmtree(subset_dir, ignore_errors=True)

    return output_dir / f"{result_name}_eval_results.json"


def build_summary_json(
    raw_result_path: Path,
    prompt_map: Dict[str, str],
    output_path: Path,
    videos_dir: Path,
    dimensions: Tuple[str, ...],
) -> None:
    """重整官方输出，生成含平均得分与逐视频得分的 JSON。"""
    raw_results = load_json(str(raw_result_path))
    summary: Dict[str, Dict[str, object]] = {}
    per_video_accumulator: Dict[str, Dict[str, float]] = {
        filename: {} for filename in prompt_map
    }

    for dimension in dimensions:
        values = raw_results.get(dimension)
        if not isinstance(values, list) or len(values) != 2:
            summary[dimension] = {
                "average_score": None,
                "videos": [],
                "note": "未获得该维度的有效输出，已原样跳过。",
            }
            continue

        average_score, video_items = values
        dimension_scores = []
        for item in video_items:
            video_name = Path(item["video_path"]).name
            if video_name not in prompt_map:
                continue
            score = float(item["video_results"])
            dimension_scores.append(
                {"video": video_name, "prompt": prompt_map[video_name], "score": score}
            )
            per_video_accumulator[video_name][dimension] = score

        summary[dimension] = {
            "average_score": float(average_score),
            "videos": dimension_scores,
        }

    per_video_summary = {}
    for video_name, dim_scores in per_video_accumulator.items():
        if not dim_scores:
            continue
        scores = list(dim_scores.values())
        per_video_summary[video_name] = {
            "prompt": prompt_map[video_name],
            "mean_score": sum(scores) / len(scores),
            "dimension_scores": dim_scores,
        }

    output_payload = {
        "metadata": {
            "videos_dir": str(videos_dir.resolve()),
            "prompt_count": len(prompt_map),
            "dimensions": list(dimensions),
            "raw_result_path": str(raw_result_path.resolve()),
        },
        "dimensions": summary,
        "per_video": per_video_summary,
    }

    save_json(output_payload, str(output_path))
    print(f"[完成] 汇总结果已写入：{output_path}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 VBench 对自定义视频集合进行评估（自定义模式）"
    )
    parser.add_argument(
        "--videos_dir",
       default='/home/zlq/diffusion/vbench_result/wan2.1/JANO_344_01504_4_6_warm6',
        type=Path,
        help="待评估视频所在文件夹，需包含 mp4/gif/png 文件。",
    )
    parser.add_argument(
        "--prompt_file",
        default='/home/zlq/diffusion/vbench_result/wan_evaluate/remain_final_prompt.txt',
        type=Path,
        help="prompt.txt 文件路径，用于筛选视频并指定文本描述。",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default='/home/zlq/diffusion/vbench_result/wan_evaluate/vbench_result/41_JANO_344_01504_4_6_warm6',
        help="评估结果输出目录（默认：vbench_result/wan_evaluate）。",
    )
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=None,
        help="汇总 JSON 的输出路径，默认位于 output_dir 中自动命名。",
    )
    parser.add_argument(
        "--full_info_json",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "vbench" / "VBench_full_info.json",
        help="VBench_full_info.json 路径，如无特殊需求保持默认即可。",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="评估结果名称前缀，默认使用 custom_ + 时间戳。",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=None,
        help="自定义维度列表，仅限 custom_input 支持的维度。不指定则使用全部默认维度。",
    )
    parser.add_argument(
        "--split_per_dimension",
        action="store_true",
        help="逐维度运行评估（类似 evaluate.sh 的做法），每个维度单独生成结果文件。",
    )
    parser.add_argument(
        "--load_ckpt_from_local",
        action="store_true",
        help="若模型权重已预先下载到 ~/.cache/vbench，启用此选项以避免重复下载。",
    )
    parser.add_argument(
        "--read_frame",
        action="store_true",
        help="部分维度可切换为逐帧读取模式，默认关闭（读取视频文件）。",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    videos_dir = args.videos_dir.resolve()
    prompt_file = args.prompt_file.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dimensions: Tuple[str, ...]
    if args.dimensions:
        dimensions = tuple(args.dimensions)
    else:
        dimensions = DEFAULT_DIMENSIONS

    raw_entries = parse_prompt_file(prompt_file)
    prompt_map = match_prompts_to_videos(videos_dir, raw_entries)

    if args.split_per_dimension:
        for dim in dimensions:
            print(f"[信息] 单独评估维度：{dim}")
            result_name = args.name or f"custom_{dim}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            raw_result_path = run_vbench_evaluation(
                videos_path=videos_dir,
                prompt_map=prompt_map,
                output_dir=output_dir,
                result_name=result_name,
                dimensions=(dim,),
                full_json_path=args.full_info_json.resolve(),
                load_ckpt_from_local=args.load_ckpt_from_local,
                read_frame=args.read_frame,
            )

            summary_path = (
                args.summary_json.resolve()
                if args.summary_json
                else output_dir / f"{result_name}_summary.json"
            )
            build_summary_json(
                raw_result_path=raw_result_path,
                prompt_map=prompt_map,
                output_path=summary_path,
                videos_dir=videos_dir,
                dimensions=(dim,),
            )
    else:
        result_name = args.name or f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        raw_result_path = run_vbench_evaluation(
            videos_path=videos_dir,
            prompt_map=prompt_map,
            output_dir=output_dir,
            result_name=result_name,
            dimensions=dimensions,
            full_json_path=args.full_info_json.resolve(),
            load_ckpt_from_local=args.load_ckpt_from_local,
            read_frame=args.read_frame,
        )

        summary_path = (
            args.summary_json.resolve()
            if args.summary_json
            else output_dir / f"{result_name}_summary.json"
        )
        build_summary_json(
            raw_result_path=raw_result_path,
            prompt_map=prompt_map,
            output_path=summary_path,
            videos_dir=videos_dir,
            dimensions=dimensions,
        )


if __name__ == "__main__":
    main()


