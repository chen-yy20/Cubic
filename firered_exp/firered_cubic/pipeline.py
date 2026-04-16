from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from firered.pipeline import (
    CONDITION_IMAGE_SIZE,
    VAE_IMAGE_SIZE,
    QwenImageEditPlusPipeline,
    calculate_shift,
    calculate_dimensions,
    retrieve_timesteps,
)
from flexflow import (
    AdaptivePartitioner,
    CubicVisualizer,
    CurvatureMonitor,
    Phase,
    SelectiveStepScheduler,
    UpdateFrequencyOptimizer,
)
from utils.timer import get_timer
from firered_cubic.cubic_selective_forward import FireRedSelectiveForwardEngine
from firered_cubic.transformer import CubicQwenImageTransformer2DModel

logger = logging.getLogger(__name__)


@dataclass
class StepSelectionStats:
    step_idx: int
    phase: str
    main_active: int
    main_total: int
    cond_active: int
    cond_total: int
    total_selected: int
    total_tokens: int
    selective_enabled: bool


class CubicQwenImageEditPlusPipeline(QwenImageEditPlusPipeline):
    """FireRed image-edit pipeline with Cubic monitoring plus selective main-token updates."""

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        processor: Qwen2VLProcessor,
        transformer: QwenImageTransformer2DModel | CubicQwenImageTransformer2DModel,
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            transformer=transformer,
        )

        self.phase_warmup_fraction = 0.15
        self.phase_cooldown_fraction = 0.15
        self.phase_anchor_interval = 8

        self.target_speedup = 2.0
        self.alpha = 2.0
        self.beta = 4.0
        self.tau_min = 1
        self.tau_max = 10

        self.patch_size = (1, 2, 2)
        self.curvature_percentile = 0.9
        self.curvature_contrast_gamma = 1.8
        self.tau_diversity_target_span = 2.0
        self.tau_diversity_max_boost = 2.5

        self.min_block_size = 2
        self.max_block_size = 16
        self.min_block_tokens = 16
        self.max_block_tokens = 256
        self.cv_threshold = 0.5
        self.max_blocksize_candidates = 36

        self.monitor = CurvatureMonitor(self)
        self.visualizer = CubicVisualizer(self)
        self.partitioner = AdaptivePartitioner(self)
        self.frequency_optimizer = UpdateFrequencyOptimizer(self)
        self.step_scheduler = SelectiveStepScheduler(self)
        self.selective_engine = FireRedSelectiveForwardEngine(self)
        self.current_partition = None
        self.current_frequency_plan = None
        self.current_step_plan = None
        self.step_selection_history: list[StepSelectionStats] = []

    def _get_inference_phase(self, step: int, total_steps: int) -> str:
        if total_steps <= 0:
            return "phase4_cooldown"

        warmup_steps = max(0, int(total_steps * self.phase_warmup_fraction))
        cooldown_steps = max(0, int(total_steps * self.phase_cooldown_fraction))
        cooldown_start = max(0, total_steps - cooldown_steps)

        if step < warmup_steps:
            return "phase1_warmup"
        if step >= cooldown_start:
            return "phase4_cooldown"

        anchor_interval = max(1, int(self.phase_anchor_interval))
        if (step - warmup_steps) % anchor_interval == 0:
            return "phase2_cruise_anchor"
        return "phase3_cruise_selective"

    def _normalize_t_to_unit_interval(self, t: torch.Tensor | float) -> float:
        if isinstance(t, torch.Tensor):
            if t.numel() == 0:
                return 0.0
            t_val = float(t.detach().reshape(-1)[0].item())
        else:
            t_val = float(t)

        if t_val > 1.0:
            scheduler_steps = float(getattr(self.scheduler.config, "num_train_timesteps", 1000) - 1)
            t_val = t_val / max(1.0, scheduler_steps)
        return max(0.0, min(1.0, t_val))

    def _sync_cubic_modules_config(self) -> None:
        self.partitioner.min_block_size = int(self.min_block_size)
        self.partitioner.max_block_size = int(self.max_block_size)
        self.partitioner.min_block_tokens = int(self.min_block_tokens)
        self.partitioner.max_block_tokens = int(self.max_block_tokens)
        self.partitioner.cv_threshold = float(self.cv_threshold)
        self.partitioner.max_candidates = int(self.max_blocksize_candidates)

        self.frequency_optimizer.target_speedup = float(self.target_speedup)
        self.frequency_optimizer.alpha = float(self.alpha)
        self.frequency_optimizer.beta = float(self.beta)
        self.frequency_optimizer.tau_min = int(self.tau_min)
        self.frequency_optimizer.tau_max = int(self.tau_max)
        self.frequency_optimizer.curvature_contrast_gamma = float(self.curvature_contrast_gamma)
        self.frequency_optimizer.tau_diversity_target_span = float(self.tau_diversity_target_span)
        self.frequency_optimizer.tau_diversity_max_boost = float(self.tau_diversity_max_boost)

        self.step_scheduler.warmup_fraction = float(self.phase_warmup_fraction)
        self.step_scheduler.cooldown_fraction = float(self.phase_cooldown_fraction)
        self.step_scheduler.anchor_interval = int(self.phase_anchor_interval)

    @staticmethod
    def _phase_to_enum(phase_name: str) -> Phase:
        if phase_name == "phase1_warmup":
            return Phase.WARMUP
        if phase_name == "phase2_cruise_anchor":
            return Phase.CRUISE_ANCHOR
        if phase_name == "phase3_cruise_selective":
            return Phase.CRUISE_SELECTIVE
        return Phase.COOLDOWN

    @torch.no_grad()
    def __call__(
        self,
        image=None,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float | None = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        cubic_warmup_fraction: float | None = None,
        cubic_cooldown_fraction: float | None = None,
        cubic_anchor_interval: int | None = None,
        cubic_target_speedup: float | None = None,
        cubic_enable_visualization: bool = True,
        cubic_results_dir: str | None = None,
        cubic_visual_tag: str | None = None,
    ):
        """Run FireRed image editing with Cubic monitoring enabled."""
        if cubic_warmup_fraction is not None:
            self.phase_warmup_fraction = float(cubic_warmup_fraction)
        if cubic_cooldown_fraction is not None:
            self.phase_cooldown_fraction = float(cubic_cooldown_fraction)
        if cubic_anchor_interval is not None:
            self.phase_anchor_interval = int(cubic_anchor_interval)
        if cubic_target_speedup is not None:
            self.target_speedup = float(cubic_target_speedup)

        self._sync_cubic_modules_config()
        self.monitor.reset()
        self.selective_engine.reset()
        self.current_partition = None
        self.current_frequency_plan = None
        self.current_step_plan = None
        self.step_selection_history = []

        image_size = image[-1].size if isinstance(image, list) else image.size
        calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
        height = height or calculated_height
        width = width or calculated_width

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if batch_size > 1:
            raise ValueError(
                f"QwenImageEditPlusPipeline currently only supports batch_size=1, but received batch_size={batch_size}. "
                "Please process prompts one at a time."
            )

        device = self._execution_device
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            if not isinstance(image, list):
                image = [image]
            condition_images = []
            vae_image_sizes = []
            vae_images = []
            for img in image:
                image_width, image_height = img.size
                condition_width, condition_height = calculate_dimensions(
                    CONDITION_IMAGE_SIZE, image_width / image_height
                )
                vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
                vae_image_sizes.append((vae_width, vae_height))
                condition_images.append(self.image_processor.resize(img, condition_height, condition_width))
                vae_images.append(self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))
        else:
            condition_images = image if isinstance(image, list) else [image]
            vae_image_sizes = []
            vae_images = image

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=condition_images,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=condition_images,
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            vae_images,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                *[
                    (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
                    for vae_width, vae_height in vae_image_sizes
                ],
            ]
        ] * batch_size

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        warmup_steps = max(0, int(len(timesteps) * self.phase_warmup_fraction))
        cooldown_steps = max(0, int(len(timesteps) * self.phase_cooldown_fraction))
        cooldown_start = max(0, len(timesteps) - cooldown_steps)

        logger.info(
            "[firered-cubic] steps=%d warmup=%d cooldown=%d cooldown_start=%d anchor_interval=%d target_speedup=%.3f gamma=%.3f diversity_span=%.3f",
            len(timesteps),
            warmup_steps,
            cooldown_steps,
            cooldown_start,
            int(self.phase_anchor_interval),
            float(self.target_speedup),
            float(self.curvature_contrast_gamma),
            float(self.tau_diversity_target_span),
        )

        self.scheduler.set_begin_index(0)
        with get_timer("generate_e2e"):
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                progress_bar.set_description("Denoising")
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    self._current_timestep = t
                    phase = self._get_inference_phase(i, len(timesteps))
                    phase_enum = self._phase_to_enum(phase)

                    if self.current_partition is not None and self.current_frequency_plan is not None:
                        selective_step_idx = max(0, i - warmup_steps)
                        self.current_step_plan = self.step_scheduler.build_step_plan(
                            partition=self.current_partition,
                            block_intervals=self.current_frequency_plan.block_intervals,
                            phase=phase_enum,
                            step_idx=i,
                            selective_step_idx=selective_step_idx,
                        )
                        step_debug = self.step_scheduler.describe_step_plan(
                            self.current_step_plan,
                            self.current_partition,
                            step_idx=i,
                        )
                        logger.info(
                            "[firered-step-plan] step=%d phase=%s active_blocks=%d frozen_blocks=%d active_tokens=%d frozen_tokens=%d full_compute=%s",
                            i,
                            step_debug.phase.value,
                            step_debug.active_block_count,
                            step_debug.frozen_block_count,
                            step_debug.active_token_count,
                            step_debug.frozen_token_count,
                            step_debug.is_full_compute,
                        )

                    selective_enabled = self.current_step_plan is not None and (not self.current_step_plan.is_full_compute)
                    step_selection = self._build_step_selection_stats(
                        step_idx=i,
                        phase=phase,
                        main_token_count=int(latents.shape[1]),
                        cond_token_count=0 if image_latents is None else int(image_latents.shape[1]),
                        step_plan=self.current_step_plan,
                    )
                    self.step_selection_history.append(step_selection)
                    logger.info(
                        "[firered-phase] step=%d/%d phase=%s cubic_selective_enabled=%s",
                        i,
                        len(timesteps) - 1,
                        phase,
                        selective_enabled,
                    )
                    logger.info(
                        "[firered-selective-step] step=%d phase=%s main_active=%d/%d cond_active=%d/%d total_selected=%d/%d selective_enabled=%s",
                        step_selection.step_idx,
                        step_selection.phase,
                        step_selection.main_active,
                        step_selection.main_total,
                        step_selection.cond_active,
                        step_selection.cond_total,
                        step_selection.total_selected,
                        step_selection.total_tokens,
                        step_selection.selective_enabled,
                    )

                    latent_model_input = latents
                    if image_latents is not None:
                        latent_model_input = torch.cat([latents, image_latents], dim=1)

                    timestep = t.expand(latents.shape[0]).to(latents.dtype)
                    with get_timer("full_forward"):
                        with self.transformer.cache_context("cond"):
                            noise_pred = self.selective_engine.forward(
                                transformer=self.transformer,
                                hidden_states=latent_model_input,
                                timestep=timestep / 1000,
                                guidance=guidance,
                                encoder_hidden_states_mask=prompt_embeds_mask,
                                encoder_hidden_states=prompt_embeds,
                                img_shapes=img_shapes,
                                attention_kwargs=self.attention_kwargs,
                                step_plan=self.current_step_plan,
                                cache_key="cond",
                                main_token_count=int(latents.shape[1]),
                                step_idx=i,
                                step_phase=phase,
                            )
                        noise_pred = noise_pred[:, : latents.size(1)]

                        if do_true_cfg:
                            with self.transformer.cache_context("uncond"):
                                neg_noise_pred = self.selective_engine.forward(
                                    transformer=self.transformer,
                                    hidden_states=latent_model_input,
                                    timestep=timestep / 1000,
                                    guidance=guidance,
                                    encoder_hidden_states_mask=negative_prompt_embeds_mask,
                                    encoder_hidden_states=negative_prompt_embeds,
                                    img_shapes=img_shapes,
                                    attention_kwargs=self.attention_kwargs,
                                    step_plan=self.current_step_plan,
                                    cache_key="uncond",
                                    main_token_count=int(latents.shape[1]),
                                    step_idx=i,
                                    step_phase=phase,
                                )
                            neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                            comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                            noise_pred = comb_pred * (cond_norm / noise_norm)

                    is_last_warmup = warmup_steps > 0 and phase == "phase1_warmup" and i == warmup_steps - 1
                    is_cruise_anchor = phase == "phase2_cruise_anchor"
                    if is_last_warmup or is_cruise_anchor:
                        with get_timer("anchor_step_monitor"):
                            latents_unpacked = self._unpack_latents(latents, height, width, self.vae_scale_factor)
                            noise_unpacked = self._unpack_latents(noise_pred, height, width, self.vae_scale_factor)
                            curvature_map = self.monitor.update(
                                x=latents_unpacked,
                                v=noise_unpacked,
                                t=self._normalize_t_to_unit_interval(t),
                                step_idx=i,
                            )

                        if curvature_map is not None:
                            logger.info(
                                "[firered-monitor] step=%d curvature_map=%s mean=%.6e min=%.6e max=%.6e",
                                i,
                                tuple(curvature_map.shape),
                                float(curvature_map.float().mean().item()),
                                float(curvature_map.float().amin().item()),
                                float(curvature_map.float().amax().item()),
                            )
                            with get_timer("anchor_step_partitioner"):
                                ft, ht, wt = curvature_map.shape
                                partition = self.partitioner.partition(
                                    Ft=ft,
                                    Ht=ht,
                                    Wt=wt,
                                    curvature_map=curvature_map,
                                )
                                partition_debug = self.partitioner.describe_partition(
                                    partition,
                                    curvature_map=curvature_map,
                                )
                                block_curvatures = self.monitor.get_block_curvatures(
                                    partition,
                                    curvature_map=curvature_map,
                                )

                            logger.info(
                                "[firered-partitioner] step=%d chosen_block_size=%s blocks=%d token_grid=(%d,%d,%d) token_stats(min=%d max=%d mean=%.2f) block_cv(mean=%.4f p90=%.4f)",
                                i,
                                partition_debug.chosen_block_size,
                                partition_debug.block_count,
                                ft,
                                ht,
                                wt,
                                partition_debug.block_token_min,
                                partition_debug.block_token_max,
                                partition_debug.block_token_mean,
                                partition_debug.mean_block_cv,
                                partition_debug.p90_block_cv,
                            )

                            with get_timer("anchor_step_optimizer"):
                                frequency_plan = self.frequency_optimizer.optimize(
                                    partition=partition,
                                    block_curvatures=block_curvatures,
                                )
                                self.current_partition = partition
                                self.current_frequency_plan = frequency_plan
                                self.current_step_plan = self.step_scheduler.build_step_plan(
                                    partition=partition,
                                    block_intervals=frequency_plan.block_intervals,
                                    phase=Phase.CRUISE_ANCHOR,
                                    step_idx=i,
                                    selective_step_idx=max(0, i - warmup_steps),
                                )

                            logger.info(
                                "[firered-optimizer] step=%d tau(min=%d max=%d mean=%.3f) mean_update_rate=%.6f achieved_speedup=%.3f histogram=%s",
                                i,
                                frequency_plan.min_interval,
                                frequency_plan.max_interval,
                                frequency_plan.mean_interval,
                                frequency_plan.mean_update_rate,
                                frequency_plan.achieved_speedup,
                                self._format_interval_histogram(frequency_plan.block_intervals),
                            )

                            if cubic_enable_visualization and cubic_results_dir:
                                self.visualizer.visualize_partition_and_frequency(
                                    partition=partition,
                                    curvature_map=curvature_map,
                                    block_curvatures=block_curvatures,
                                    block_intervals=frequency_plan.block_intervals,
                                    step_idx=i,
                                    output_dir=cubic_results_dir,
                                    file_suffix=cubic_visual_tag,
                                )
                                logger.info(
                                    "[firered-visual] step=%d partition visualization exported to %s/partition_vis",
                                    i,
                                    cubic_results_dir,
                                )
                        else:
                            logger.info(
                                "[firered-monitor] step=%d first anchor collected; waiting for next anchor to form drift-based curvature.",
                                i,
                            )

                    latents_dtype = latents.dtype
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if cubic_enable_visualization and cubic_results_dir and self.monitor.curvature_history:
            self.visualizer.export_curvature_history(
                self.monitor.curvature_history,
                cubic_results_dir,
                file_suffix=cubic_visual_tag,
            )
            logger.info(
                "[firered-visual] curvature history exported to %s (records=%d)",
                cubic_results_dir,
                len(self.monitor.curvature_history),
            )
        self._log_step_selection_summary()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)

    @staticmethod
    def _format_interval_histogram(block_intervals: dict[int, int]) -> str:
        if not block_intervals:
            return "{}"
        counts = Counter(int(v) for v in block_intervals.values())
        return "{" + ", ".join(f"{tau}:{counts[tau]}" for tau in sorted(counts)) + "}"

    @staticmethod
    def _build_step_selection_stats(
        step_idx: int,
        phase: str,
        main_token_count: int,
        cond_token_count: int,
        step_plan,
    ) -> StepSelectionStats:
        main_total = max(0, int(main_token_count))
        cond_total = max(0, int(cond_token_count))
        total_tokens = main_total + cond_total

        if step_plan is None or step_plan.is_full_compute:
            main_active = main_total
            cond_active = cond_total
            selective_enabled = False
        else:
            main_active = min(main_total, int(step_plan.active_token_indices.numel()))
            cond_active = cond_total
            selective_enabled = True

        total_selected = main_active + cond_active
        return StepSelectionStats(
            step_idx=int(step_idx),
            phase=str(phase),
            main_active=main_active,
            main_total=main_total,
            cond_active=cond_active,
            cond_total=cond_total,
            total_selected=total_selected,
            total_tokens=total_tokens,
            selective_enabled=selective_enabled,
        )

    def _log_step_selection_summary(self) -> None:
        if not self.step_selection_history:
            return

        total_steps = len(self.step_selection_history)
        selective_steps = sum(1 for item in self.step_selection_history if item.selective_enabled)
        main_active_sum = sum(item.main_active for item in self.step_selection_history)
        main_total_sum = sum(item.main_total for item in self.step_selection_history)
        cond_active_sum = sum(item.cond_active for item in self.step_selection_history)
        cond_total_sum = sum(item.cond_total for item in self.step_selection_history)
        total_selected_sum = sum(item.total_selected for item in self.step_selection_history)
        total_tokens_sum = sum(item.total_tokens for item in self.step_selection_history)

        logger.info(
            "[firered-selective-summary] steps=%d selective_steps=%d main_active_avg=%.2f/%d cond_active_avg=%.2f/%d total_selected_avg=%.2f/%d avg_selected_ratio=%.4f",
            total_steps,
            selective_steps,
            main_active_sum / max(1, total_steps),
            self.step_selection_history[0].main_total,
            cond_active_sum / max(1, total_steps),
            self.step_selection_history[0].cond_total,
            total_selected_sum / max(1, total_steps),
            self.step_selection_history[0].total_tokens,
            total_selected_sum / max(1, total_tokens_sum),
        )

        phase_totals: dict[str, list[StepSelectionStats]] = {}
        for item in self.step_selection_history:
            phase_totals.setdefault(item.phase, []).append(item)

        for phase_name, items in phase_totals.items():
            logger.info(
                "[firered-selective-summary] phase=%s steps=%d main_active_avg=%.2f/%d cond_active_avg=%.2f/%d total_selected_avg=%.2f/%d avg_selected_ratio=%.4f",
                phase_name,
                len(items),
                sum(item.main_active for item in items) / max(1, len(items)),
                items[0].main_total,
                sum(item.cond_active for item in items) / max(1, len(items)),
                items[0].cond_total,
                sum(item.total_selected for item in items) / max(1, len(items)),
                items[0].total_tokens,
                sum(item.total_selected for item in items) / max(1, sum(item.total_tokens for item in items)),
            )
