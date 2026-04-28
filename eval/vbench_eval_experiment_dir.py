import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from vbench import VBench


DEFAULT_CUSTOM_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]

VIDEO_EXTS = (".mp4", ".gif", ".mov", ".webm", ".avi", ".mkv")


def _log(msg: str) -> None:
    print(f"[vbench_exp_eval] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_single_params_json(exp_dir: Path) -> Path:
    matches = sorted(exp_dir.glob("*_params_metrics.json"))
    if not matches:
        raise FileNotFoundError(f"在目录中未找到 *_params_metrics.json: {exp_dir}")
    if len(matches) > 1:
        raise RuntimeError(
            f"目录里有多个 *_params_metrics.json，无法唯一确定: {matches}\n"
            "请用 --params_json 指定。"
        )
    return matches[0]


def _find_video(exp_dir: Path) -> Path:
    videos = sorted([p for p in exp_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
    if not videos:
        raise FileNotFoundError(
            f"在目录中没有找到视频文件（支持后缀: {VIDEO_EXTS}）: {exp_dir}\n"
            "请通过 --video_path 显式指定视频文件路径。"
        )
    if len(videos) > 1:
        raise RuntimeError(f"目录里有多个视频文件，无法唯一确定: {videos}\n请用 --video_path 指定。")
    return videos[0]


def _parse_dims(raw: str) -> List[str]:
    dims = [d.strip() for d in raw.split(",") if d.strip()]
    if not dims:
        raise ValueError("--dimensions 不能为空")
    return dims


def main() -> None:
    parser = argparse.ArgumentParser(description="适配 Cubic 实验目录结构的 VBench 评测脚本")
    parser.add_argument("--exp_dir", required=True, help="实验目录，如 results/wan/1.3B/cubic/xxx")
    parser.add_argument("--video_path", default=None, help="视频路径（不填则自动在 exp_dir 下找唯一视频）")
    parser.add_argument("--params_json", default=None, help="参数 JSON 路径（不填则自动在 exp_dir 下找）")
    parser.add_argument(
        "--dimensions",
        default=",".join(DEFAULT_CUSTOM_DIMENSIONS),
        help="custom_input 维度，逗号分隔",
    )
    parser.add_argument("--device", default="cuda", help="cuda / cpu")
    parser.add_argument("--vbench_info_path", default="VBench_full_info.json", help="VBench_full_info.json 路径")
    parser.add_argument("--output_dir", default="evaluation_results/vbench_custom", help="VBench 输出目录")
    parser.add_argument("--name", default=None, help="本次评测名（不填自动生成）")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir).expanduser().resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"exp_dir 不存在: {exp_dir}")

    params_json = Path(args.params_json).expanduser().resolve() if args.params_json else _find_single_params_json(exp_dir)
    if not params_json.exists():
        raise FileNotFoundError(f"params_json 不存在: {params_json}")

    video_path = Path(args.video_path).expanduser().resolve() if args.video_path else _find_video(exp_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"video_path 不存在: {video_path}")

    vbench_info_path = Path(args.vbench_info_path).expanduser().resolve()
    if not vbench_info_path.exists():
        raise FileNotFoundError(f"未找到 VBench_full_info.json: {vbench_info_path}")

    dimensions = _parse_dims(args.dimensions)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    params_obj = _read_json(params_json)
    prompt = params_obj.get("parameters", {}).get("prompt")
    run_name = args.name or f"{exp_dir.name}_custom_{time.strftime('%Y%m%d_%H%M%S')}"

    _log(f"exp_dir={exp_dir}")
    _log(f"video_path={video_path}")
    _log(f"params_json={params_json}")
    _log(f"prompt={prompt}")
    _log(f"dimensions={dimensions}")

    _log("初始化 VBench ...")
    evaluator = VBench(args.device, str(vbench_info_path), str(output_dir))

    _log("开始 VBench custom_input 评测 ...")
    t0 = time.perf_counter()
    evaluator.evaluate(
        videos_path=str(video_path),
        name=run_name,
        dimension_list=dimensions,
        mode="custom_input",
    )
    elapsed = time.perf_counter() - t0
    _log(f"评测完成，耗时 {elapsed:.1f}s")

    quality_metrics = params_obj.get("quality_metrics")
    if not isinstance(quality_metrics, dict):
        quality_metrics = {}
    quality_metrics["vbench"] = {
        "run_name": run_name,
        "mode": "custom_input",
        "video_path": str(video_path),
        "dimensions": dimensions,
        "device": args.device,
        "vbench_output_dir": str(output_dir),
        "elapsed_seconds": round(elapsed, 2),
    }
    params_obj["quality_metrics"] = quality_metrics
    _write_json(params_json, params_obj)
    _log(f"已回写评测记录到: {params_json}")


if __name__ == "__main__":
    main()
