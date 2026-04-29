"""Flux2-klein cubic runtime for straightness evaluation.

This module hosts cubic-specific execution logic so that the entry script keeps
the same role as origin generate scripts: init args and call into runtime.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cubic_core import (
    AdaptivePartitioner,
    CurvatureMonitor,
    CubicVisualizer,
    Phase,
    StraightnessEvaluator,
)
from flux2_klein.pipeline import Flux2KleinPipeline, compute_empirical_mu, retrieve_timesteps
from flux2_klein.transformer import Flux2Transformer2DModel
from .cubic_selective_forward import Flux2KSelectiveForwardEngine
from .optimizer import Flux2KAdaptiveOptimizer
from utils.results import get_prompt_id
from utils.timer import get_timer

TAG = "cubic"

# 初始化 cubic selective forward engine（全局单例，后续用于 selective step）
cubic_selective_engine = Flux2KSelectiveForwardEngine(config=None)


def resolve_output_paths(
    model_path: str,
    prompt: str,
    output_dir: Path | None,
    output_image: Path | None,
    mode_suffix: str = "null",
) -> tuple[Path, Path, str]:
    model_id = Path(model_path).name
    prompt_id = get_prompt_id(prompt)
    mode_suffix = str(mode_suffix).lower()
    resolved_dir = output_dir or Path("results") / "flux2" / model_id / f"{TAG}_{mode_suffix}" / prompt_id
    resolved_name = f"{TAG}_{prompt_id}_{model_id}.png"
    resolved_image = output_image or (resolved_dir / resolved_name)
    return resolved_dir, resolved_image, model_id


def load_pipeline(model_path: str, dtype: torch.dtype, device: str) -> Flux2KleinPipeline:
    print(f"Loading custom Flux2Klein pipeline from {model_path}...", flush=True)
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    pipe = Flux2KleinPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        transformer=transformer,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    print(f"Pipeline ready on device={pipe._execution_device}", flush=True)
    return pipe


def run_straightness_sampling(
    pipe: Flux2KleinPipeline,
    prompt: str,
    height: int,
    width: int,
    guidance_scale: float,
    num_inference_steps: int,
    generator: torch.Generator,
    max_sequence_length: int,
    cubic_results_dir: Path | None = None,
    enable_step_timing: bool = True,
) -> tuple[object, StraightnessEvaluator]:
    with torch.inference_mode():
        # flux2k专用优化：
        # - anchor前两步固定full
        # - 其余步骤为selective阶段
        # - 每个partition仅分配cooldown步数 c_b，表示“最后c_b步active”
        cubic_cfg = SimpleNamespace(
            patch_size=(1, 1, 1),
            curvature_percentile=0.9,
            partition_gpu_aggregation=True,
            min_block_size=2,
            max_block_size=16,
            min_block_tokens=16,
            max_block_tokens=256,
            cv_threshold=0.5,
            max_blocksize_candidates=36,
            flux2k_max_cooldown_steps=2,
        )
        monitor = CurvatureMonitor(cubic_cfg)
        partitioner = AdaptivePartitioner(cubic_cfg)
        optimizer = Flux2KAdaptiveOptimizer(cubic_cfg)
        visualizer = CubicVisualizer(cubic_cfg)

        current_partition = None
        current_policy_plan = None

        prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_embeds=None,
            device=pipe._execution_device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )

        negative_prompt_embeds = None
        negative_text_ids = None
        if guidance_scale > 1.0 and not pipe.config.is_distilled:
            negative_prompt_embeds, negative_text_ids = pipe.encode_prompt(
                prompt="",
                prompt_embeds=None,
                device=pipe._execution_device,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
            )

        num_channels_latents = pipe.transformer.config.in_channels // 4
        latents, latent_ids = pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=pipe._execution_device,
            generator=generator,
            latents=None,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if getattr(pipe.scheduler.config, "use_flow_sigmas", False):
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler,
            num_inference_steps,
            pipe._execution_device,
            sigmas=sigmas,
            mu=mu,
        )
        pipe.scheduler.set_begin_index(0)

        straightness_cfg = SimpleNamespace(
            patch_size=(1, 1, 1),
            straightness_time_eps=1e-3,
            straightness_space_eps=1e-3,
        )
        straightness = StraightnessEvaluator(straightness_cfg)

        monitor.reset()
        cubic_selective_engine.reset()

        # 前两步anchor（full），后两步selective（按step_plan selective执行）
        anchor_steps = 2
        with pipe.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                step_timer_name = f"step_{i:04d}_total"
                step_timer_ctx = get_timer(step_timer_name) if enable_step_timing else nullcontext()
                with step_timer_ctx:
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)
                    phase = Phase.CRUISE_ANCHOR if i < anchor_steps else Phase.CRUISE_SELECTIVE
                    phase_label = phase.name.lower().replace("cruise_", "")
                    step_plan = None

                    if phase == Phase.CRUISE_SELECTIVE and current_partition is not None and current_policy_plan is not None:
                        step_plan = optimizer.build_step_plan(
                            partition=current_partition,
                            plan=current_policy_plan,
                            step_idx=i,
                        )
                        if step_plan is None:
                            phase = Phase.COOLDOWN
                            phase_label = phase.name.lower()

                    print(
                        f"[cubic][step={i:02d}][{phase_label}] begin t={float(t.detach().reshape(-1)[0].item()):.6f}",
                        flush=True,
                    )

                    with pipe.transformer.cache_context("cond"):
                        noise_pred = cubic_selective_engine.forward(
                            transformer=pipe.transformer,
                            hidden_states=latents.to(pipe.transformer.dtype),
                            timestep=timestep / 1000,
                            guidance=None,
                            pooled_projections=None,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_ids,
                            joint_attention_kwargs=None,
                            step_plan=step_plan,
                            cache_key="flux2k_cond",
                            step_idx=i,
                            step_phase=phase_label,
                            trace_label="cond",
                        )

                    if negative_prompt_embeds is not None and negative_text_ids is not None:
                        with pipe.transformer.cache_context("uncond"):
                            neg_noise_pred = cubic_selective_engine.forward(
                                transformer=pipe.transformer,
                                hidden_states=latents.to(pipe.transformer.dtype),
                                timestep=timestep / 1000,
                                guidance=None,
                                pooled_projections=None,
                                encoder_hidden_states=negative_prompt_embeds,
                                txt_ids=negative_text_ids,
                                img_ids=latent_ids,
                                joint_attention_kwargs=None,
                                step_plan=step_plan,
                                cache_key="flux2k_uncond",
                                step_idx=i,
                                step_phase=phase_label,
                                trace_label="uncond",
                            )
                        noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

                    latents_unpacked = pipe._unpack_latents_with_ids(latents, latent_ids)
                    noise_unpacked = pipe._unpack_latents_with_ids(noise_pred, latent_ids)

                    if phase == Phase.CRUISE_ANCHOR:
                        curvature_map = monitor.update(
                            x=latents_unpacked.unsqueeze(2),
                            v=noise_unpacked.unsqueeze(2),
                            t=normalize_t_to_unit_interval(t, pipe.scheduler),
                            step_idx=i,
                        )
                        if curvature_map is not None:
                            ft, ht, wt = curvature_map.shape
                            current_partition = partitioner.partition(
                                Ft=ft,
                                Ht=ht,
                                Wt=wt,
                                curvature_map=curvature_map,
                            )
                            block_curvatures = monitor.get_block_curvatures(
                                current_partition,
                                curvature_map=curvature_map,
                            )
                            current_policy_plan = optimizer.optimize(
                                partition=current_partition,
                                block_curvatures=block_curvatures,
                                total_steps=len(timesteps),
                                anchor_steps=anchor_steps,
                            )
                            if current_policy_plan.cooldown_by_block:
                                cooldown_values = list(current_policy_plan.cooldown_by_block.values())
                                cd_min = min(cooldown_values)
                                cd_max = max(cooldown_values)
                                cd_mean = sum(cooldown_values) / len(cooldown_values)
                            else:
                                cd_min = cd_max = 0
                                cd_mean = 0.0
                            print(
                                f"[cubic][step={i:02d}][anchor] plan "
                                f"blocks={len(current_partition.blocks)} "
                                f"selective_steps={current_policy_plan.selective_steps} "
                                f"cooldown=[{cd_min},{cd_max}] mean={cd_mean:.2f}",
                                flush=True,
                            )
                            if cubic_results_dir is not None:
                                visualizer.visualize_partition_and_frequency(
                                    partition=current_partition,
                                    curvature_map=curvature_map,
                                    block_curvatures=block_curvatures,
                                    block_intervals=current_policy_plan.cooldown_by_block,
                                    step_idx=i,
                                    output_dir=str(cubic_results_dir),
                                    block_value_label='c',
                                )

                    if step_plan is not None:
                        total_block_count = len(current_partition.blocks)
                        active_block_count = len(step_plan.active_block_ids)
                        print(
                            f"[cubic][step={i:02d}][{phase_label}] plan "
                            f"active_blocks={active_block_count}/{total_block_count} "
                            f"active_tokens={int(step_plan.active_token_indices.numel())}/{current_partition.total_tokens} "
                            f"full_compute={step_plan.is_full_compute}",
                            flush=True,
                        )

                    record = straightness.update(
                        x=latents_unpacked.unsqueeze(2),
                        v=noise_unpacked.unsqueeze(2),
                        t=normalize_t_to_unit_interval(t, pipe.scheduler),
                        step_idx=i,
                    )
                    if record is not None:
                        print(
                            f"[cubic][step={int(record.step):02d}][straightness] "
                            f"t={record.t:.4f} mean_ratio={record.mean_ratio:.6f} corr={record.corr:.6f}",
                            flush=True,
                        )

                    latents_dtype = latents.dtype
                    latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                    progress_bar.update()

        latents = pipe._unpack_latents_with_ids(latents, latent_ids)
        latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = pipe._unpatchify_latents(latents)
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]

    return image, straightness


def normalize_t_to_unit_interval(t: torch.Tensor | float, scheduler) -> float:
    if isinstance(t, torch.Tensor):
        if t.numel() == 0:
            return 0.0
        t_val = float(t.detach().reshape(-1)[0].item())
    else:
        t_val = float(t)

    if t_val > 1.0:
        scheduler_steps = float(getattr(scheduler.config, "num_train_timesteps", 1000) - 1)
        t_val = t_val / max(1.0, scheduler_steps)
    return max(0.0, min(1.0, t_val))


def save_straightness_summary(output_dir: Path, evaluator: StraightnessEvaluator) -> Path:
    summary_path = output_dir / "straightness_summary.txt"
    lines: list[str] = []
    lines.append("# Straightness Summary")
    lines.append(f"record_count={len(evaluator.records)}")
    lines.append("step,t,dt,mean_drift,mean_proxy_curvature,mean_curvature_gt,mean_ratio,corr")
    for rec in evaluator.records:
        lines.append(
            f"{rec.step},{rec.t:.6f},{rec.dt:.6f},{rec.mean_drift:.6e},"
            f"{rec.mean_proxy_curvature:.6e},{rec.mean_curvature_gt:.6e},"
            f"{rec.mean_ratio:.6e},{rec.corr:.6f}"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path
