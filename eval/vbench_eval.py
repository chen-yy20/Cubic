import argparse
import json
import time
from pathlib import Path
from typing import List

from vbench import VBench


ALL_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "appearance_style",
    "overall_consistency",
]

# 官方文档里明确支持 custom_input 的维度
CUSTOM_INPUT_SUPPORTED_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]


def _log(msg: str) -> None:
    print(f"[vbench_eval] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def _parse_dimensions(raw: str) -> List[str]:
    token = raw.strip().lower()
    if token == "all":
        return ALL_DIMENSIONS
    dims = [d.strip() for d in raw.split(",") if d.strip()]
    if not dims:
        raise ValueError("`--dimension` 不能为空。可用示例：human_action 或 subject_consistency,aesthetic_quality")
    invalid = [d for d in dims if d not in ALL_DIMENSIONS]
    if invalid:
        raise ValueError(f"存在非法维度: {invalid}。支持维度为: {ALL_DIMENSIONS}")
    return dims


def _auto_name(videos_path: Path, dimensions: List[str], mode: str) -> str:
    dim_part = dimensions[0] if len(dimensions) == 1 else f"{len(dimensions)}dims"
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{videos_path.name}_{mode}_{dim_part}_{ts}"


def main() -> None:
    parser = argparse.ArgumentParser(description="VBench 评测脚本（基于 VBench 官方 Python API）")
    parser.add_argument("--videos_path", required=True, help="待评测视频文件或视频目录路径")
    parser.add_argument(
        "--dimension",
        required=True,
        help="维度名，逗号分隔多维度，或 all（16 维）",
    )
    parser.add_argument(
        "--mode",
        default="vbench_standard",
        choices=["vbench_standard", "custom_input", "vbench_category"],
        help="评测模式：标准基准 / 自定义输入 / 按类别评测",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="仅 mode=vbench_category 时需要；类别名例如 animals、humans 等",
    )
    parser.add_argument(
        "--vbench_info_path",
        default="VBench_full_info.json",
        help="VBench_full_info.json 路径（默认当前目录下）",
    )
    parser.add_argument(
        "--output_dir",
        default="evaluation_results",
        help="结果输出目录",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="本次评测名字（不填则自动生成）",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="运行设备（如 cuda / cpu）",
    )
    args = parser.parse_args()

    videos_path = Path(args.videos_path).expanduser().resolve()
    if not videos_path.exists():
        raise FileNotFoundError(f"videos_path 不存在: {videos_path}")

    vbench_info_path = Path(args.vbench_info_path).expanduser().resolve()
    if not vbench_info_path.exists():
        raise FileNotFoundError(
            f"未找到 VBench 信息文件: {vbench_info_path}\n"
            "请先下载 VBench_full_info.json，或通过 --vbench_info_path 指定其位置。"
        )

    dimensions = _parse_dimensions(args.dimension)

    if args.mode == "custom_input":
        not_supported = [d for d in dimensions if d not in CUSTOM_INPUT_SUPPORTED_DIMENSIONS]
        if not_supported:
            raise ValueError(
                "custom_input 模式仅支持以下维度："
                f"{CUSTOM_INPUT_SUPPORTED_DIMENSIONS}；当前不支持: {not_supported}"
            )

    if args.mode == "vbench_category" and not args.category:
        raise ValueError("mode=vbench_category 时必须提供 --category")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.name or _auto_name(videos_path, dimensions, args.mode)

    _log("开始初始化 VBench ...")
    t0 = time.perf_counter()
    evaluator = VBench(args.device, str(vbench_info_path), str(output_dir))
    _log(f"VBench 初始化完成，耗时 {time.perf_counter() - t0:.1f}s")

    eval_kwargs = {
        "videos_path": str(videos_path),
        "name": run_name,
        "dimension_list": dimensions,
        "mode": args.mode,
    }
    if args.mode == "vbench_category":
        eval_kwargs["category"] = args.category

    _log(f"开始评测: name={run_name}, mode={args.mode}, dimensions={dimensions}")
    t1 = time.perf_counter()
    evaluator.evaluate(**eval_kwargs)
    elapsed = time.perf_counter() - t1
    _log(f"评测完成，耗时 {elapsed:.1f}s")

    manifest = {
        "name": run_name,
        "mode": args.mode,
        "videos_path": str(videos_path),
        "dimension_list": dimensions,
        "category": args.category,
        "output_dir": str(output_dir),
        "vbench_info_path": str(vbench_info_path),
        "device": args.device,
        "elapsed_seconds": round(elapsed, 2),
    }
    manifest_path = output_dir / f"{run_name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"已写入运行清单: {manifest_path}")


if __name__ == "__main__":
    main()
