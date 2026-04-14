from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from flux.pipeline import FluxPipeline, calculate_shift, retrieve_timesteps
from flexflow import AdaptivePartitioner, CurvatureMonitor, CubicVisualizer, UpdateFrequencyOptimizer
from utils.timer import get_timer

from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from flux.transformer import FluxTransformer2DModel


class CubicFluxPipeline(FluxPipeline):
    """Flux pipeline with Cubic stage classification + anchor monitoring."""

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
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

        # Flux images tend to have smoother global curvature; amplify contrast for tau mapping.
        self.curvature_contrast_gamma = 1.8

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

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        cubic_warmup_fraction: Optional[float] = None,
        cubic_cooldown_fraction: Optional[float] = None,
        cubic_anchor_interval: Optional[int] = None,
        cubic_target_speedup: Optional[float] = None,
        cubic_enable_visualization: bool = True,
        cubic_results_dir: Optional[str] = None,
        cubic_visual_tag: Optional[str] = None,
    ):
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

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        self._num_timesteps = len(timesteps)

        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        warmup_steps = max(0, int(len(timesteps) * self.phase_warmup_fraction))

        with get_timer("generate_e2e"):
            pbar = tqdm(
                timesteps,
                desc="Denoising",
                dynamic_ncols=True,
                position=0,
                leave=True,
            )
            for i, t in enumerate(pbar):
                if self.interrupt:
                    continue

                phase = self._get_inference_phase(i, len(timesteps))

                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                is_last_warmup = warmup_steps > 0 and phase == "phase1_warmup" and i == warmup_steps - 1
                is_cruise_anchor = phase == "phase2_cruise_anchor"
                if is_last_warmup or is_cruise_anchor:
                    with get_timer("anchor_step_monitor"):
                        latents_unpacked = self._unpack_latents(latents, height, width, self.vae_scale_factor)
                        noise_unpacked = self._unpack_latents(noise_pred, height, width, self.vae_scale_factor)
                        curvature_map = self.monitor.update(
                            x=latents_unpacked.unsqueeze(2),
                            v=noise_unpacked.unsqueeze(2),
                            t=self._normalize_t_to_unit_interval(t),
                            step_idx=i,
                        )

                    if curvature_map is not None:
                        with get_timer("anchor_step_partitioner"):
                            ft, ht, wt = curvature_map.shape
                            partition = self.partitioner.partition(
                                Ft=ft,
                                Ht=ht,
                                Wt=wt,
                                curvature_map=curvature_map,
                            )
                            block_curvatures = self.monitor.get_block_curvatures(
                                partition,
                                curvature_map=curvature_map,
                            )

                        with get_timer("anchor_step_optimizer"):
                            frequency_plan = self.frequency_optimizer.optimize(
                                partition=partition,
                                block_curvatures=block_curvatures,
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

                pbar.update(0)

            pbar.close()

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if cubic_enable_visualization and cubic_results_dir and self.monitor.curvature_history:
            self.visualizer.export_curvature_history(
                self.monitor.curvature_history,
                cubic_results_dir,
                file_suffix=cubic_visual_tag,
            )

        if not return_dict:
            return (image,)

        from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

        return FluxPipelineOutput(images=image)

    def _sync_cubic_modules_config(self) -> None:
        # Partitioner configs
        self.partitioner.min_block_size = int(self.min_block_size)
        self.partitioner.max_block_size = int(self.max_block_size)
        self.partitioner.min_block_tokens = int(self.min_block_tokens)
        self.partitioner.max_block_tokens = int(self.max_block_tokens)
        self.partitioner.cv_threshold = float(self.cv_threshold)
        self.partitioner.max_candidates = int(self.max_blocksize_candidates)

        # Optimizer configs
        self.frequency_optimizer.target_speedup = float(self.target_speedup)
        self.frequency_optimizer.alpha = float(self.alpha)
        self.frequency_optimizer.beta = float(self.beta)
        self.frequency_optimizer.tau_min = int(self.tau_min)
        self.frequency_optimizer.tau_max = int(self.tau_max)
        self.frequency_optimizer.curvature_contrast_gamma = float(self.curvature_contrast_gamma)
