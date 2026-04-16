from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers.models.attention_dispatch import dispatch_attention_fn

REPO_ROOT = Path(__file__).resolve().parents[2]
FIRERED_EXP_ROOT = Path(__file__).resolve().parents[1]
for extra_path in (REPO_ROOT, FIRERED_EXP_ROOT):
    extra_str = str(extra_path)
    if extra_str not in sys.path:
        sys.path.insert(0, extra_str)

from firered.pipeline import QwenImageEditPlusPipeline
from firered.transformer import QwenImageTransformer2DModel, apply_rotary_emb_qwen
from utils.results import get_prompt_id

MODEL_PATH = "/home/chenyy/WORK/models/firered-image-edit"
IMAGE_PATH = "/home/chenyy/WORK/cyy/cubic/firered_exp/examples/cola.png"
PROMPT = "Transform the object into a realistic miniature product by carefully holding it between your thumb and forefinger."
TAG = "cond_kv_drift"

logger = logging.getLogger(__name__)


def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
        force=True,
    )


@dataclass
class ForwardProbeContext:
    step_idx: int
    timestep: float
    branch: str | None
    main_token_count: int
    cond_token_count: int


@dataclass
class LayerDriftRecord:
    step_idx: int
    timestep: float
    branch: str
    layer_idx: int
    main_token_count: int
    cond_token_count: int
    cond_k_mse: float | None
    cond_v_mse: float | None
    cond_kv_mse: float | None
    main_k_mse: float | None
    main_v_mse: float | None
    main_kv_mse: float | None
    cond_to_main_ratio: float | None


class CondKVDriftRecorder:
    def __init__(self, sampled_layers: Iterable[int]):
        self.sampled_layers = sorted({int(layer_idx) for layer_idx in sampled_layers})
        self.current_context: ForwardProbeContext | None = None
        self.records: list[LayerDriftRecord] = []
        self.prev_cond_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.prev_main_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.cond_step_counter = 0

    def begin_forward(
        self,
        branch: str | None,
        timestep: torch.Tensor | float | None,
        img_shapes,
    ) -> None:
        if not img_shapes:
            self.current_context = None
            return

        sample_shapes = img_shapes[0]
        if not sample_shapes:
            self.current_context = None
            return

        main_token_count = int(math.prod(sample_shapes[0]))
        cond_token_count = int(sum(math.prod(shape) for shape in sample_shapes[1:]))
        if branch == "cond":
            step_idx = self.cond_step_counter
            self.cond_step_counter += 1
        else:
            step_idx = max(0, self.cond_step_counter - 1)

        self.current_context = ForwardProbeContext(
            step_idx=step_idx,
            timestep=self._to_float_timestep(timestep),
            branch=branch,
            main_token_count=main_token_count,
            cond_token_count=cond_token_count,
        )

    def end_forward(self) -> None:
        self.current_context = None

    def record_layer(self, layer_idx: int, img_key: torch.Tensor, img_value: torch.Tensor) -> None:
        context = self.current_context
        if context is None or context.branch != "cond" or context.cond_token_count <= 0:
            return

        main_end = context.main_token_count
        cond_start = context.main_token_count
        cond_end = cond_start + context.cond_token_count
        if cond_end > int(img_key.shape[1]):
            raise ValueError(
                f"Condition token slice [{cond_start}:{cond_end}] exceeds image KV sequence length {int(img_key.shape[1])}."
            )

        main_key = img_key[:, :main_end].detach().float().cpu()
        main_value = img_value[:, :main_end].detach().float().cpu()
        cond_key = img_key[:, cond_start:cond_end].detach().float().cpu()
        cond_value = img_value[:, cond_start:cond_end].detach().float().cpu()

        prev_main_key, prev_main_value = self.prev_main_kv.get(layer_idx, (None, None))
        prev_key, prev_value = self.prev_cond_kv.get(layer_idx, (None, None))

        main_k_mse, main_v_mse, main_kv_mse = self._compute_kv_mse(
            current_key=main_key,
            current_value=main_value,
            prev_key=prev_main_key,
            prev_value=prev_main_value,
        )
        cond_k_mse, cond_v_mse, cond_kv_mse = self._compute_kv_mse(
            current_key=cond_key,
            current_value=cond_value,
            prev_key=prev_key,
            prev_value=prev_value,
        )

        cond_to_main_ratio = None
        if cond_kv_mse is not None and main_kv_mse is not None and main_kv_mse > 0.0:
            cond_to_main_ratio = cond_kv_mse / main_kv_mse

        self.records.append(
            LayerDriftRecord(
                step_idx=context.step_idx,
                timestep=context.timestep,
                branch=context.branch,
                layer_idx=int(layer_idx),
                main_token_count=context.main_token_count,
                cond_token_count=context.cond_token_count,
                cond_k_mse=cond_k_mse,
                cond_v_mse=cond_v_mse,
                cond_kv_mse=cond_kv_mse,
                main_k_mse=main_k_mse,
                main_v_mse=main_v_mse,
                main_kv_mse=main_kv_mse,
                cond_to_main_ratio=cond_to_main_ratio,
            )
        )
        self.prev_main_kv[layer_idx] = (main_key, main_value)
        self.prev_cond_kv[layer_idx] = (cond_key, cond_value)

    @staticmethod
    def _compute_kv_mse(
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        prev_key: torch.Tensor | None,
        prev_value: torch.Tensor | None,
    ) -> tuple[float | None, float | None, float | None]:
        if prev_key is None or prev_value is None or prev_key.shape != current_key.shape or prev_value.shape != current_value.shape:
            return None, None, None
        k_mse = float(torch.mean((current_key - prev_key) ** 2).item())
        v_mse = float(torch.mean((current_value - prev_value) ** 2).item())
        return k_mse, v_mse, 0.5 * (k_mse + v_mse)

    @staticmethod
    def _to_float_timestep(timestep: torch.Tensor | float | None) -> float:
        if timestep is None:
            return float("nan")
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 0:
                return float("nan")
            return float(timestep.detach().reshape(-1)[0].item())
        return float(timestep)


class KVProbeQwenDoubleStreamAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, recorder: CondKVDriftRecorder, layer_idx: int):
        self.recorder = recorder
        self.layer_idx = int(layer_idx)

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        del encoder_hidden_states_mask

        if encoder_hidden_states is None:
            raise ValueError("KV probe processor requires encoder_hidden_states (text stream).")

        seq_txt = encoder_hidden_states.shape[1]

        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        self.recorder.record_layer(self.layer_idx, img_key=img_key, img_value=img_value)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)

        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())

        return img_attn_output, txt_attn_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe FireRed origin cond-image KV drift across timesteps.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--input_image", type=Path, nargs="+", default=[Path(IMAGE_PATH)])
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument(
        "--sample_layers",
        type=str,
        default="0,5,10,20,30,40,50,55,last",
        help="Comma-separated layer indices. Supports `last`.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def load_pipeline(model_path: str) -> QwenImageEditPlusPipeline:
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
    pipe.set_progress_bar_config(disable=None)
    return pipe


def resolve_sampled_layers(transformer: QwenImageTransformer2DModel, spec: str) -> list[int]:
    num_layers = len(transformer.transformer_blocks)
    resolved: list[int] = []
    for raw_item in spec.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if item == "last":
            resolved.append(num_layers - 1)
            continue
        layer_idx = int(item)
        if layer_idx < 0:
            layer_idx = num_layers + layer_idx
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"Layer index {layer_idx} is out of range for num_layers={num_layers}.")
        resolved.append(layer_idx)
    if not resolved:
        raise ValueError("No valid sample layers were provided.")
    
    return sorted(set(resolved))


def install_kv_probe(transformer: QwenImageTransformer2DModel, recorder: CondKVDriftRecorder, sampled_layers: list[int]) -> None:
    for layer_idx in sampled_layers:
        transformer.transformer_blocks[layer_idx].attn.processor = KVProbeQwenDoubleStreamAttnProcessor(
            recorder=recorder,
            layer_idx=layer_idx,
        )

    original_cache_context = transformer.cache_context

    @contextmanager
    def cache_context_with_probe(name: str):
        previous_branch = getattr(transformer, "_kv_probe_branch", None)
        transformer._kv_probe_branch = name
        with original_cache_context(name):
            yield
        transformer._kv_probe_branch = previous_branch

    transformer.cache_context = cache_context_with_probe  # type: ignore[method-assign]

    original_forward = transformer.forward

    def forward_with_probe(*args, **kwargs):
        branch = getattr(transformer, "_kv_probe_branch", None)
        timestep = kwargs.get("timestep")
        img_shapes = kwargs.get("img_shapes")
        recorder.begin_forward(branch=branch, timestep=timestep, img_shapes=img_shapes)
        try:
            return original_forward(*args, **kwargs)
        finally:
            recorder.end_forward()

    transformer.forward = forward_with_probe  # type: ignore[method-assign]


def export_records(output_dir: Path, records: list[LayerDriftRecord]) -> Path:
    csv_path = output_dir / "cond_kv_drift_records.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return csv_path


def export_summary(output_dir: Path, args: argparse.Namespace, sampled_layers: list[int], records: list[LayerDriftRecord]) -> Path:
    valid_cond_records = [record for record in records if record.cond_kv_mse is not None]
    valid_main_records = [record for record in records if record.main_kv_mse is not None]
    valid_pair_records = [record for record in records if record.cond_to_main_ratio is not None]
    by_layer: dict[int, list[LayerDriftRecord]] = {}
    for record in records:
        by_layer.setdefault(record.layer_idx, []).append(record)

    layer_summary = {}
    for layer_idx, items in by_layer.items():
        cond_items = [item for item in items if item.cond_kv_mse is not None]
        main_items = [item for item in items if item.main_kv_mse is not None]
        ratio_items = [item for item in items if item.cond_to_main_ratio is not None]

        layer_summary[str(layer_idx)] = {
            "samples": len(items),
            "valid_cond_samples": len(cond_items),
            "valid_main_samples": len(main_items),
            "valid_ratio_samples": len(ratio_items),
            "mean_cond_kv_mse": (
                sum(item.cond_kv_mse for item in cond_items if item.cond_kv_mse is not None) / len(cond_items)
                if cond_items
                else None
            ),
            "mean_main_kv_mse": (
                sum(item.main_kv_mse for item in main_items if item.main_kv_mse is not None) / len(main_items)
                if main_items
                else None
            ),
            "mean_cond_to_main_ratio": (
                sum(item.cond_to_main_ratio for item in ratio_items if item.cond_to_main_ratio is not None) / len(ratio_items)
                if ratio_items
                else None
            ),
            "max_cond_to_main_ratio": (
                max(item.cond_to_main_ratio for item in ratio_items if item.cond_to_main_ratio is not None)
                if ratio_items
                else None
            ),
            "min_cond_to_main_ratio": (
                min(item.cond_to_main_ratio for item in ratio_items if item.cond_to_main_ratio is not None)
                if ratio_items
                else None
            ),
        }

    summary = {
        "tag": TAG,
        "parameters": {
            "model": "firered",
            "method": "origin_probe",
            "model_path": args.model_path,
            "input_image": [str(path) for path in args.input_image],
            "prompt": args.prompt,
            "seed": args.seed,
            "true_cfg_scale": args.true_cfg_scale,
            "num_inference_steps": args.num_inference_steps,
            "sample_layers": sampled_layers,
        },
        "layer_summary": layer_summary,
        "record_count": len(records),
        "valid_cond_record_count": len(valid_cond_records),
        "valid_main_record_count": len(valid_main_records),
        "valid_ratio_record_count": len(valid_pair_records),
    }
    summary_path = output_dir / f"{TAG}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary_path


def export_plot(output_dir: Path, records: list[LayerDriftRecord], sampled_layers: list[int]) -> Path:
    step_map: dict[int, dict[str, Any]] = {}
    for record in records:
        bucket = step_map.setdefault(
            record.step_idx,
            {
                "timestep": record.timestep,
                "cond": [],
                "main": [],
                "ratio": [],
            },
        )
        if record.cond_kv_mse is not None:
            bucket["cond"].append(record.cond_kv_mse)
        if record.main_kv_mse is not None:
            bucket["main"].append(record.main_kv_mse)
        if record.cond_to_main_ratio is not None:
            bucket["ratio"].append(record.cond_to_main_ratio)

    ordered_steps = sorted(step_map.keys())
    if not ordered_steps:
        raise ValueError("No drift records were collected; cannot plot KV drift comparison.")

    timesteps = [step_map[step_idx]["timestep"] for step_idx in ordered_steps]
    cond_mean = [
        float(sum(step_map[step_idx]["cond"]) / len(step_map[step_idx]["cond"])) if step_map[step_idx]["cond"] else float("nan")
        for step_idx in ordered_steps
    ]
    main_mean = [
        float(sum(step_map[step_idx]["main"]) / len(step_map[step_idx]["main"])) if step_map[step_idx]["main"] else float("nan")
        for step_idx in ordered_steps
    ]
    ratio_mean = [
        float(sum(step_map[step_idx]["ratio"]) / len(step_map[step_idx]["ratio"])) if step_map[step_idx]["ratio"] else float("nan")
        for step_idx in ordered_steps
    ]

    valid_layer_records = [
        record for record in records if record.cond_kv_mse is not None and record.main_kv_mse is not None
    ]
    layer_cond_mean: list[float] = []
    layer_main_mean: list[float] = []
    for layer_idx in sampled_layers:
        layer_records = [record for record in valid_layer_records if record.layer_idx == layer_idx]
        if not layer_records:
            continue
        layer_cond_mean.append(sum(record.cond_kv_mse for record in layer_records if record.cond_kv_mse is not None) / len(layer_records))
        layer_main_mean.append(sum(record.main_kv_mse for record in layer_records if record.main_kv_mse is not None) / len(layer_records))

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=False)

    axes[0].plot(timesteps, cond_mean, marker="o", linewidth=1.8, label="cond KV drift")
    axes[0].plot(timesteps, main_mean, marker="s", linewidth=1.8, label="latent(main) KV drift")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Mean KV MSE vs previous step")
    axes[0].set_title("Per-step mean KV drift (across sampled layers)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].invert_xaxis()

    axes[1].plot(timesteps, ratio_mean, marker="d", color="tab:red", linewidth=1.8)
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Mean cond/main KV drift ratio")
    axes[1].set_title("Per-step cond/main drift ratio")
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_xaxis()

    axes[2].scatter(layer_main_mean, layer_cond_mean, alpha=0.75, s=36)
    if layer_main_mean and layer_cond_mean:
        min_val = min(min(layer_main_mean), min(layer_cond_mean))
        max_val = max(max(layer_main_mean), max(layer_cond_mean))
        axes[2].plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", linewidth=1.2)
    axes[2].set_xlabel("Layer mean latent(main) KV drift")
    axes[2].set_ylabel("Layer mean cond KV drift")
    axes[2].set_title("Layer-level cond vs latent(main) drift")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("FireRed origin cond vs latent(main) KV drift comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    figure_path = output_dir / "cond_vs_latent_kv_drift_plot.png"
    plt.savefig(figure_path, dpi=180)
    plt.close()
    return figure_path


def resolve_output_dir(model_path: str, prompt: str, output_dir: Path | None) -> Path:
    model_id = Path(model_path).name
    prompt_id = get_prompt_id(prompt)
    return output_dir or Path("results") / "firered" / model_id / TAG / prompt_id


def main() -> None:
    _init_logging()
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this FireRed KV drift experiment.")

    output_dir = resolve_output_dir(args.model_path, args.prompt, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading FireRed origin pipeline from %s", args.model_path)
    pipeline = load_pipeline(args.model_path)
    sampled_layers = resolve_sampled_layers(pipeline.transformer, args.sample_layers)
    recorder = CondKVDriftRecorder(sampled_layers=sampled_layers)
    install_kv_probe(pipeline.transformer, recorder=recorder, sampled_layers=sampled_layers)
    logger.info("Sampling cond-image KV drift from layers=%s", sampled_layers)

    images = [Image.open(path).convert("RGB") for path in args.input_image]
    inputs = {
        "image": images,
        "prompt": args.prompt,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": " ",
        "num_inference_steps": args.num_inference_steps,
        "num_images_per_prompt": 1,
        "output_type": "latent",
    }

    with torch.inference_mode():
        pipeline(**inputs)

    if not recorder.records:
        raise RuntimeError("No KV drift records were collected. Verify sampled layers and cond-image inputs.")

    csv_path = export_records(output_dir, recorder.records)
    summary_path = export_summary(output_dir, args=args, sampled_layers=sampled_layers, records=recorder.records)
    figure_path = export_plot(output_dir, recorder.records, sampled_layers=sampled_layers)

    logger.info("Cond KV drift records saved to %s", csv_path)
    logger.info("Cond KV drift summary saved to %s", summary_path)
    logger.info("Cond KV drift plot saved to %s", figure_path)


if __name__ == "__main__":
    main()
