"""
在 PIE-Bench++（parquet）上批量运行 FireRed-Image-Edit，支持多种加速/方法：

  ori, teacache, jano, pab, cubic

与各自 `*_generate.py` 的管线一致；数据与提示词规则见 pie_bench_common.py。

用法（在 Cubic 根目录）:
  python firered_exp/pie_bench_batch_generate.py --method teacache --parquet data/pie_++/...
  python launch.py --model firered --method pie_bench -- --method jano --skip_existing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

_CUBIC_ROOT = Path(__file__).resolve().parent.parent
_FIRERED_EXP = Path(__file__).resolve().parent
for _p in (_CUBIC_ROOT, _FIRERED_EXP):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from pie_bench_common import (
    _parse_edit_action,
    edit_action_to_natural_language,
    image_bytes_to_pil,
    target_prompt_to_hpsv3_caption,
)
from utils.launch_defaults import resolve_model_path
from utils.results import save_params_and_metrics

DEFAULT_PARQUET = str(_CUBIC_ROOT / "data/pie_++/V1-00000-of-00001.parquet")

PIE_METHODS = ("ori", "teacache", "jano", "pab", "cubic")


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


# ── 条件 import：与对应 generate 一致 ─────────────────────────────────


def _monkey_import(method: str) -> None:
    if method == "teacache":
        import baseline.teacache_firered  # noqa: F401
    elif method == "pab":
        import baseline.PAB_firered  # noqa: F401
    elif method == "jano":
        import baseline.jano_firered  # noqa: F401


def load_pipeline(
    model_path: str, method: str, disable_progress_bar: bool = True
) -> Any:
    _monkey_import(method)
    if method == "cubic":
        from firered_cubic.pipeline import CubicQwenImageEditPlusPipeline
        from firered_cubic.transformer import CubicQwenImageTransformer2DModel

        pipe = CubicQwenImageEditPlusPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            transformer=CubicQwenImageTransformer2DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
            ),
        )
    else:
        from firered.pipeline import QwenImageEditPlusPipeline
        from firered.transformer import QwenImageTransformer2DModel

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
    pipe.set_progress_bar_config(
        disable=True if disable_progress_bar else None
    )
    return pipe


def setup_transformer(
    method: str, pipeline: Any, args: argparse.Namespace
) -> None:
    """按方法配置 transformer；cubic 无需此步。"""
    if method == "cubic":
        return
    tr = pipeline.transformer
    if method == "teacache":
        tr.enable_teacache = True
        tr.num_steps = args.num_inference_steps
        tr.rel_l1_thresh = args.rel_l1_thresh
        tr.teacache_cache_device = next(tr.parameters()).device
        if hasattr(tr, "reset_teacache_state"):
            tr.reset_teacache_state()
    elif method == "pab":
        pab_cooldown_steps = args.num_inference_steps - args.pab_cooldown_steps_offset
        tr.enable_pab = True
        tr.num_steps = args.num_inference_steps
        tr.pab_warmup_steps = args.pab_warmup_steps
        tr.pab_cooldown_steps = pab_cooldown_steps
        tr.pab_skip_range = args.pab_skip_range
        if hasattr(tr, "reset_pab_state"):
            tr.reset_pab_state()
    elif method == "jano":
        block = (1, args.jano_analyze_block_h, args.jano_analyze_block_w)
        tr.enable_jano = True
        tr.num_steps = args.num_inference_steps
        tr.jano_warmup_steps = args.jano_warmup_steps
        tr.jano_cooldown_steps = args.jano_cooldown_steps
        tr.jano_static_interval = args.jano_static_interval
        tr.jano_medium_interval = args.jano_medium_interval
        tr.jano_static_thresh = args.jano_static_thresh
        tr.jano_medium_thresh = args.jano_medium_thresh
        tr.jano_ref_thresh = args.jano_ref_thresh
        tr.jano_analyze_block_size = block
        if hasattr(tr, "reset_jano_state"):
            tr.reset_jano_state()
    # ori: 不 patch


def reset_method_state(method: str, pipeline: Any) -> None:
    """每条样本前重置有状态的加速模块，避免跨图污染。"""
    if method == "cubic":
        return
    tr = pipeline.transformer
    if method == "teacache" and hasattr(tr, "reset_teacache_state"):
        tr.reset_teacache_state()
    elif method == "pab" and hasattr(tr, "reset_pab_state"):
        tr.reset_pab_state()
    elif method == "jano" and hasattr(tr, "reset_jano_state"):
        tr.reset_jano_state()


def build_model_inputs(
    method: str,
    pil: Image.Image,
    prompt: str,
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "image": [pil],
        "prompt": prompt,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": " ",
        "num_inference_steps": args.num_inference_steps,
        "num_images_per_prompt": 1,
    }
    if method == "cubic":
        inputs.update(
            {
                "cubic_target_speedup": args.cubic_target_speedup,
                "cubic_cond_target_speedup": args.cubic_cond_target_speedup,
                "cubic_warmup_fraction": args.cubic_warmup_fraction,
                "cubic_cooldown_fraction": args.cubic_cooldown_fraction,
                "cubic_anchor_interval": args.cubic_anchor_interval,
                "cubic_enable_visualization": args.cubic_enable_visualization,
                "cubic_results_dir": str(run_dir),
                "cubic_visual_tag": f"seed{args.seed}",
            }
        )
    return inputs


def method_params_extra(method: str, args: argparse.Namespace) -> dict[str, Any]:
    """写入 JSON 的方法专属字段。"""
    d: dict[str, Any] = {"acceleration": method}
    if method == "teacache":
        d["rel_l1_thresh"] = args.rel_l1_thresh
    elif method == "pab":
        d["pab_warmup_steps"] = args.pab_warmup_steps
        d["pab_cooldown_steps"] = args.num_inference_steps - args.pab_cooldown_steps_offset
        d["pab_skip_range"] = args.pab_skip_range
    elif method == "jano":
        d.update(
            {
                "jano_warmup_steps": args.jano_warmup_steps,
                "jano_cooldown_steps": args.jano_cooldown_steps,
                "jano_static_interval": args.jano_static_interval,
                "jano_medium_interval": args.jano_medium_interval,
                "jano_static_thresh": args.jano_static_thresh,
                "jano_medium_thresh": args.jano_medium_thresh,
                "jano_ref_thresh": args.jano_ref_thresh,
                "jano_analyze_block": [
                    1,
                    args.jano_analyze_block_h,
                    args.jano_analyze_block_w,
                ],
            }
        )
    elif method == "cubic":
        d.update(
            {
                "cubic_target_speedup": args.cubic_target_speedup,
                "cubic_cond_target_speedup": args.cubic_cond_target_speedup,
                "cubic_warmup_fraction": args.cubic_warmup_fraction,
                "cubic_cooldown_fraction": args.cubic_cooldown_fraction,
                "cubic_anchor_interval": args.cubic_anchor_interval,
                "cubic_enable_visualization": args.cubic_enable_visualization,
            }
        )
    return d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch FireRed on PIE-Bench++ (ori / teacache / jano / pab / cubic)"
    )
    p.add_argument(
        "--method",
        type=str,
        default="ori",
        choices=PIE_METHODS,
        help="与 firered_exp 下各 generate 入口对应；输出子目录名默认与此相同",
    )
    p.add_argument(
        "--parquet",
        type=Path,
        default=Path(DEFAULT_PARQUET),
    )
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="默认 results/firered/{model}/pie_bench",
    )
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help="结果子目录名；默认与 --method 相同",
    )
    p.add_argument("--seed", type=int, default=49)
    p.add_argument("--true_cfg_scale", type=float, default=4.0)
    p.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="未指定时：cubic=40，其余=50（与各 generate 默认一致）",
    )
    p.add_argument(
        "--edit_lead_in",
        type=str,
        default="Edit the image: ",
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--ids", type=str, default=None)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--fail_fast", action="store_true")
    p.add_argument(
        "--disable_progress_bar", action="store_true", default=True
    )
    # teacache
    p.add_argument(
        "--rel_l1_thresh",
        type=float,
        default=0.2,
        help="[teacache] 同 teacache_generate",
    )
    # pab
    p.add_argument("--pab_warmup_steps", type=int, default=5)
    p.add_argument("--pab_cooldown_steps_offset", type=int, default=5)
    p.add_argument("--pab_skip_range", type=int, default=4)
    # jano
    p.add_argument("--jano_warmup_steps", type=int, default=5)
    p.add_argument("--jano_cooldown_steps", type=int, default=5)
    p.add_argument("--jano_static_interval", type=int, default=6)
    p.add_argument("--jano_medium_interval", type=int, default=3)
    p.add_argument("--jano_static_thresh", type=float, default=0.2)
    p.add_argument("--jano_medium_thresh", type=float, default=0.6)
    p.add_argument("--jano_ref_thresh", type=float, default=0.5)
    p.add_argument("--jano_analyze_block_h", type=int, default=4)
    p.add_argument("--jano_analyze_block_w", type=int, default=4)
    # cubic
    p.add_argument("--cubic_target_speedup", type=float, default=8.0)
    p.add_argument("--cubic_cond_target_speedup", type=float, default=8.0)
    p.add_argument("--cubic_warmup_fraction", type=float, default=0.1)
    p.add_argument("--cubic_cooldown_fraction", type=float, default=0.1)
    p.add_argument("--cubic_anchor_interval", type=int, default=8)
    p.add_argument(
        "--cubic_enable_visualization",
        action="store_true",
        default=False,
        help="[cubic] 批量时默认关曲率可视化以省盘；需可视化再加此开关",
    )
    return p.parse_args()


def main() -> None:
    _init_logging()
    args = parse_args()
    args.model_path = resolve_model_path(args.model_path, "--model_path")
    if args.num_inference_steps is None:
        args.num_inference_steps = 40 if args.method == "cubic" else 50
    tag = args.tag or args.method

    pq_path = args.parquet
    if not pq_path.is_file():
        raise FileNotFoundError(f"parquet not found: {pq_path.resolve()}")

    table = pq.read_table(pq_path)
    n = table.num_rows
    col_names = set(table.column_names)
    required = {"image", "id", "source_prompt", "target_prompt", "edit_action"}
    if not required.issubset(col_names):
        raise ValueError(
            f"parquet 缺少列: {required - col_names}. 实际: {col_names}"
        )
    d = table.to_pydict()
    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        indices = [i for i in range(n) if d["id"][i] in want]
        if not indices:
            raise ValueError("没有匹配的 --ids")
    else:
        e = args.end if args.end is not None else n
        indices = list(range(args.start, min(e, n)))

    model_id = Path(args.model_path).name
    out_root = args.output_root or (
        Path("results") / model_id / "pie_bench"
    )
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    logging.info(
        "PIE-Bench 批量 method=%s tag=%s rows=%d job=%d out=%s",
        args.method,
        tag,
        n,
        len(indices),
        out_root,
    )

    logging.info("Loading pipeline (method=%s)…", args.method)
    pipeline = load_pipeline(
        args.model_path, args.method, args.disable_progress_bar
    )
    setup_transformer(args.method, pipeline, args)
    logging.info("Pipeline ready.", )

    done = 0
    mextra = method_params_extra(args.method, args)
    for j, i in enumerate(indices):
        sample_id = str(d["id"][i])
        source_prompt = d["source_prompt"][i] or ""
        target_prompt = d["target_prompt"][i] or ""
        hpsv3_prompt = target_prompt_to_hpsv3_caption(target_prompt)
        raw_edit = d["edit_action"][i]
        if isinstance(raw_edit, str) and not raw_edit.strip():
            logging.warning("%s: empty edit_action, skip", sample_id)
            if args.fail_fast:
                raise ValueError(f"{sample_id}: empty edit_action")
            continue
        try:
            edit_parsed = _parse_edit_action(raw_edit)
        except json.JSONDecodeError as e:
            logging.warning("%s: bad edit_action JSON: %s", sample_id, e)
            if args.fail_fast:
                raise
            continue
        prompt = edit_action_to_natural_language(
            edit_parsed, lead_in=(args.edit_lead_in or "")
        )
        if not prompt:
            logging.warning(
                "%s: empty natural language from edit_action, skip", sample_id
            )
            if args.fail_fast:
                raise ValueError(
                    f"{sample_id}: empty natural language from edit_action"
                )
            continue

        run_dir = out_root / tag / sample_id
        out_name = f"{tag}_pie_{sample_id}_{model_id}.png"
        out_path = run_dir / out_name
        if args.skip_existing and out_path.is_file():
            logging.info("[%d/%d] skip existing %s", j + 1, len(indices), out_path)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        reset_method_state(args.method, pipeline)
        try:
            pil = image_bytes_to_pil(d["image"][i])
        except Exception as ex:
            logging.exception("%s: load image failed: %s", sample_id, ex)
            if args.fail_fast:
                raise
            continue

        inputs = build_model_inputs(
            args.method, pil, prompt, run_dir, args
        )
        t0 = time.perf_counter()
        _sync_cuda()
        try:
            with torch.inference_mode():
                result = pipeline(**inputs)
        except Exception as ex:
            logging.exception("%s: inference failed: %s", sample_id, ex)
            if args.fail_fast:
                raise
            continue
        _sync_cuda()
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0

        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(out_path)
        logging.info("[%d/%d] %s -> %s", j + 1, len(indices), sample_id, out_path)
        done += 1

        extra: dict[str, Any] = {
            "parquet": str(pq_path),
            "row_index": i,
            "dataset_id": sample_id,
            "source_prompt": source_prompt,
            "target_prompt": target_prompt,
            "hpsv3_prompt": hpsv3_prompt,
            "edit_action": edit_parsed,
            "natural_language_edit_prompt": prompt,
        }
        for key in ("aspect_mapping", "blended_words", "mask"):
            if key in d and i < len(d[key]):
                val = d[key][i]
                if isinstance(val, str):
                    try:
                        extra[key] = (
                            json.loads(val)
                            if val.strip().startswith(("{", "["))
                            else val
                        )
                    except json.JSONDecodeError:
                        extra[key] = val
                else:
                    extra[key] = val

        params: dict[str, Any] = {
            "model": "firered",
            "method": f"{tag}_pie_bench",
            "model_id": model_id,
            "model_path": args.model_path,
            "input_mode": "parquet_image_bytes",
            "output_image": str(out_path),
            "edit_prompt": prompt,
            "hpsv3_prompt": hpsv3_prompt,
            "seed": args.seed,
            "true_cfg_scale": args.true_cfg_scale,
            "num_inference_steps": args.num_inference_steps,
            "pie_bench": extra,
        }
        params.update(mextra)
        save_params_and_metrics(
            str(run_dir),
            tag,
            params,
            {"inference_e2e_ms": elapsed_ms, "per_sample": True},
            quality_metrics=None,
        )

    logging.info("Done. 成功 %d / 计划 %d 条。", done, len(indices))


if __name__ == "__main__":
    main()
