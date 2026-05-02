# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import random
import sys
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist
from PIL import Image

import gc
from contextlib import contextmanager
import torchvision.transforms.functional as TF
import torch.cuda.amp as amp
import numpy as np
import math
import wan
from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from tqdm import tqdm
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import cache_video, cache_image, str2bool

# Cubic utilities for timing and quality evaluation
_WAN_EXP_ROOT = os.path.dirname(os.path.abspath(__file__))
_CUBIC_ROOT = os.path.dirname(_WAN_EXP_ROOT)
sys.path.insert(0, _CUBIC_ROOT)
from utils.quality_metric import evaluate_quality_with_origin
from utils.results import get_prompt_id, save_params_and_metrics
from utils.timer import get_time_statistics_dict, init_timer, print_time_statistics, save_time_stats_to_file, get_timer

init_timer()

# Constants for PAB
TAG = 'pab'
MODEL_PATH = "/home/fit/zhaijdcyy/WORK/models/Wan2.1-T2V-1.3B"
PROMPT = 'Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.'

# PAB Configuration
PAB_SELF_RANGE = 2  # Self-attention broadcast interval
PAB_CROSS_RANGE = 5  # Cross-attention broadcast interval
PAB_WARMUP = 5  # Warmup steps before PAB starts
PAB_COOLDOWN = 5  # Cooldown steps at the end (full computation)
PAB_LAYER_INTERVAL = 1  # Apply to every layer

model_id = '1.3B' if '1.3B' in MODEL_PATH else '14B'
OUTPUT_DIR = f'./results/wan/{model_id}/pab/{get_prompt_id(PROMPT)}'

REQUIRED_TIMER_NAMES = [
    'generate_e2e',
    'model_forward',
]

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}


# ==============================================================================
# PAB Manager Implementation
# ==============================================================================

class PabManager:
    """Pyramid Attention Broadcast Manager for Wan"""

    def __init__(self, num_inference_steps, self_range, cross_range, warmup, cooldown, layer_interval=1):
        self.warmup = warmup
        self.cooldown = cooldown
        self.layer_interval = layer_interval
        self.num_inference_steps = num_inference_steps
        self.self_range = self_range
        self.cross_range = cross_range

        self.self_calc = True
        self.cross_calc = True
        self.should_store = False

        self.self_attn_cache = {}
        self.cross_attn_cache = {}

    def check_calc(self, step):
        """Check whether to calculate or reuse attention for current step"""
        # Self attention logic
        if step < self.warmup or step > self.num_inference_steps - self.cooldown:
            self.self_calc = True
            self.should_store = False
        else:
            self.self_calc = ((step - self.warmup) % self.self_range == 0)
            self.should_store = True

        # Cross attention logic
        if step < self.warmup or step > self.num_inference_steps - self.cooldown:
            self.cross_calc = True
        else:
            self.cross_calc = ((step - self.warmup) % self.cross_range == 0)

        print(f"step {step} | PAB: self: {self.self_calc}, cross: {self.cross_calc}, store: {self.should_store}", flush=True)

    def is_target_layer(self, layer_id):
        """Check if this layer should participate in PAB"""
        return layer_id % self.layer_interval == 0

    def get_cache_key(self, cfg_flag, layer_id):
        """Generate cache key for attention storage"""
        return f"{cfg_flag}-{layer_id}"


# Global PAB manager instance
_pab_manager = None

def init_pab_manager(num_steps, self_range, cross_range, warmup, cooldown, layer_interval=1):
    """Initialize global PAB manager"""
    global _pab_manager
    _pab_manager = PabManager(num_steps, self_range, cross_range, warmup, cooldown, layer_interval)
    return _pab_manager

def get_pab_manager():
    """Get global PAB manager instance"""
    global _pab_manager
    return _pab_manager


def wrap_model_with_pab(wan_t2v, args):
    """Wrap Wan model with PAB functionality"""
    # Initialize PAB manager
    num_steps = args.sample_steps
    init_pab_manager(
        num_steps=num_steps,
        self_range=args.pab_self_range,
        cross_range=args.pab_cross_range,
        warmup=args.pab_warmup,
        cooldown=args.pab_cooldown,
        layer_interval=args.pab_layer_interval
    )

    # Replace generate method with PAB-aware version
    wan_t2v.__class__.generate = t2v_generate_pab

    # Monkey patch attention modules
    _patch_attention_modules(wan_t2v.model)


def _patch_attention_modules(model):
    """Patch attention modules with PAB logic"""
    from wan.modules.attention import flash_attention
    from utils.envs import GlobalEnv

    for layer_id, block in enumerate(model.blocks):
        # Patch self attention
        original_self_attn_forward = block.self_attn.forward

        def make_self_attn_forward(orig_fn, layer_id):
            def pab_self_attn_forward(x, seq_lens, grid_sizes, freqs):
                pab_manager = get_pab_manager()
                cfg_flag = GlobalEnv.get_envs("pab_cfg_flag") if GlobalEnv.get_envs("pab_cfg_flag") is not None else 0

                # Check if we can reuse cached attention
                if pab_manager and not pab_manager.self_calc and pab_manager.is_target_layer(layer_id):
                    cache_key = pab_manager.get_cache_key(cfg_flag, layer_id)
                    if cache_key in pab_manager.self_attn_cache:
                        print(f"PAB: Reusing self-attn cache for layer {layer_id}, cfg={cfg_flag}", flush=True)
                        return pab_manager.self_attn_cache[cache_key]

                # Compute attention
                result = orig_fn(x, seq_lens, grid_sizes, freqs)

                # Store in cache if needed
                if pab_manager and pab_manager.should_store and pab_manager.is_target_layer(layer_id):
                    cache_key = pab_manager.get_cache_key(cfg_flag, layer_id)
                    pab_manager.self_attn_cache[cache_key] = result

                return result
            return pab_self_attn_forward

        block.self_attn.forward = make_self_attn_forward(original_self_attn_forward, layer_id)

        # Patch cross attention
        if hasattr(block, 'cross_attn'):
            original_cross_attn_forward = block.cross_attn.forward

            def make_cross_attn_forward(orig_fn, layer_id):
                def pab_cross_attn_forward(x, context, context_lens):
                    pab_manager = get_pab_manager()
                    cfg_flag = GlobalEnv.get_envs("pab_cfg_flag") if GlobalEnv.get_envs("pab_cfg_flag") is not None else 0

                    # Check if we can reuse cached attention
                    if pab_manager and not pab_manager.cross_calc and pab_manager.is_target_layer(layer_id):
                        cache_key = pab_manager.get_cache_key(cfg_flag, layer_id)
                        if cache_key in pab_manager.cross_attn_cache:
                            print(f"PAB: Reusing cross-attn cache for layer {layer_id}, cfg={cfg_flag}", flush=True)
                            return pab_manager.cross_attn_cache[cache_key]

                    # Compute attention
                    result = orig_fn(x, context, context_lens)

                    # Store in cache if needed
                    if pab_manager and pab_manager.should_store and pab_manager.is_target_layer(layer_id):
                        cache_key = pab_manager.get_cache_key(cfg_flag, layer_id)
                        pab_manager.cross_attn_cache[cache_key] = result

                    return result
                return pab_cross_attn_forward

            block.cross_attn.forward = make_cross_attn_forward(original_cross_attn_forward, layer_id)


def t2v_generate_pab(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
    """PAB-aware T2V generation"""
    from utils.envs import GlobalEnv

    # Preprocess
    F = frame_num
    target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                    size[1] // self.vae_stride[1],
                    size[0] // self.vae_stride[2])

    seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (self.patch_size[1] * self.patch_size[2]) *
                        target_shape[1] / self.sp_size) * self.sp_size

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)

    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device('cpu'))
        context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        context = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]

    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g)
    ]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, 'no_sync', noop_no_sync)

    # Get PAB manager
    pab_manager = get_pab_manager()

    # Evaluation mode
    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        # Sample videos
        latents = noise

        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}

        with get_timer('generate_e2e'):
            for step, t in enumerate(tqdm(timesteps)):
                # Update PAB state for current step
                if pab_manager:
                    pab_manager.check_calc(step)

                latent_model_input = latents
                timestep = [t]
                timestep = torch.stack(timestep)

                self.model.to(self.device)

                # CFG: conditional
                GlobalEnv.set_envs("pab_cfg_flag", 0)
                with get_timer('model_forward'):
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0]

                # CFG: unconditional
                GlobalEnv.set_envs("pab_cfg_flag", 1)
                with get_timer('model_forward'):
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null)[0]

                # Combine predictions
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]

                latents = [temp_x0.squeeze(0)]

        x0 = latents
        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()
        if self.rank == 0:
            videos = self.vae.decode(x0)

    del noise, latents
    del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return videos[0] if self.rank == 0 else None


def i2v_generate_pab(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
    """PAB-aware I2V generation"""
    from utils.envs import GlobalEnv

    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

    F = frame_num
    h, w = img.shape[1:]
    aspect_ratio = h / w
    lat_h = round(
        np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
        self.patch_size[1] * self.patch_size[1])
    lat_w = round(
        np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
        self.patch_size[2] * self.patch_size[2])
    h = lat_h * self.vae_stride[1]
    w = lat_w * self.vae_stride[2]

    max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
        self.patch_size[1] * self.patch_size[2])
    max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)
    noise = torch.randn(
        self.vae.model.z_dim,
        (F - 1) // self.vae_stride[0] + 1,
        lat_h,
        lat_w,
        dtype=torch.float32,
        generator=seed_g,
        device=self.device)

    msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
    msk[:, 1:] = 0
    msk = torch.concat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
    ],
                       dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt

    # Preprocess
    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device('cpu'))
        context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        context = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]

    self.clip.model.to(self.device)
    clip_context = self.clip.visual([img[:, None, :, :]])
    if offload_model:
        self.clip.model.cpu()

    y = self.vae.encode([
        torch.concat([
            torch.nn.functional.interpolate(
                img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                    0, 1),
            torch.zeros(3, F-1, h, w)
        ],
                     dim=1).to(self.device)
    ])[0]
    y = torch.concat([msk, y])

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, 'no_sync', noop_no_sync)

    # Get PAB manager
    pab_manager = get_pab_manager()

    # Evaluation mode
    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        # Sample videos
        latent = noise

        arg_c = {
            'context': [context[0]],
            'clip_fea': clip_context,
            'seq_len': max_seq_len,
            'y': [y],
        }

        arg_null = {
            'context': context_null,
            'clip_fea': clip_context,
            'seq_len': max_seq_len,
            'y': [y],
        }

        if offload_model:
            torch.cuda.empty_cache()

        self.model.to(self.device)

        with get_timer('generate_e2e'):
            for step, t in enumerate(tqdm(timesteps)):
                # Update PAB state for current step
                if pab_manager:
                    pab_manager.check_calc(step)

                latent_model_input = [latent.to(self.device)]
                timestep = [t]
                timestep = torch.stack(timestep).to(self.device)

                # CFG: conditional
                GlobalEnv.set_envs("pab_cfg_flag", 0)
                with get_timer('model_forward'):
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0].to(
                            torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()

                # CFG: unconditional
                GlobalEnv.set_envs("pab_cfg_flag", 1)
                with get_timer('model_forward'):
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null)[0].to(
                            torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()

                # Combine predictions
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent.to(self.device)]
                del latent_model_input, timestep

        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()

        if self.rank == 0:
            videos = self.vae.decode(x0)

    del noise, latent
    del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return videos[0] if self.rank == 0 else None


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan with PAB"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="ch",
        choices=["ch", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    # PAB specific arguments
    parser.add_argument(
        "--pab_self_range",
        type=int,
        default=PAB_SELF_RANGE,
        help="Self-attention broadcast interval (skip range)."
    )
    parser.add_argument(
        "--pab_cross_range",
        type=int,
        default=PAB_CROSS_RANGE,
        help="Cross-attention broadcast interval (skip range)."
    )
    parser.add_argument(
        "--pab_warmup",
        type=int,
        default=PAB_WARMUP,
        help="Warmup steps before PAB starts."
    )
    parser.add_argument(
        "--pab_cooldown",
        type=int,
        default=PAB_COOLDOWN,
        help="Cooldown steps at the end (full computation)."
    )
    parser.add_argument(
        "--pab_layer_interval",
        type=int,
        default=PAB_LAYER_INTERVAL,
        help="Apply PAB to every N-th layer."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results.")
    parser.add_argument(
        "--ori_baseline_root",
        type=str,
        default=None,
        help=(
            "Ori 结果根目录（含各 prompt_id 子目录）。设置后与对应 prompt 的 ori 视频比对质量；"
            "用于输出目录名为 pab_s*_c_* 等无法自动推断的情形。"
        ),
    )

    args = parser.parse_args()
    # Set default values
    args.task = 't2v-1.3B' if '1.3B' in MODEL_PATH else 't2v-14B'
    args.size = '832*480' if '1.3B' in MODEL_PATH else '1280*720'
    if args.prompt is None:
        args.prompt = PROMPT
    args.ckpt_dir = MODEL_PATH
    args.base_seed = 42 if args.base_seed < 0 else args.base_seed

    _validate_args(args)

    return args


def _init_logging(rank):
    # Logging
    if rank == 0:
        # Set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    # Initialize output directory
    output_dir = args.output_dir if hasattr(args, 'output_dir') and args.output_dir else OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    if args.use_prompt_extend:
        # Delayed import to avoid dashscope dependency when not using prompt extension
        from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model, is_vl="i2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    quality_metrics = None

    if "t2v" in args.task or "t2i" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        logging.info(f"Input prompt: {args.prompt}")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        # Wrap model with PAB
        wrap_model_with_pab(wan_t2v, args)

        logging.info(
            f"Generating {'image' if 't2i' in args.task else 'video'} with PAB ...")
        with get_timer('generate_e2e'):
            video = wan_t2v.generate(
                args.prompt,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model)

    else:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]["image"]
        logging.info(f"Input prompt: {args.prompt}")
        logging.info(f"Input image: {args.image}")

        img = Image.open(args.image).convert("RGB")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    image=img,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        # Wrap model with PAB
        wrap_model_with_pab(wan_i2v, args)

        logging.info("Generating video with PAB ...")
        with get_timer('generate_e2e'):
            video = wan_i2v.generate(
                args.prompt,
                img,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model)

    if rank == 0:
        # Use tag-based filename in output directory
        filename = f'{TAG}_{get_prompt_id(args.prompt)}_{args.task}.mp4'
        os.makedirs(output_dir, exist_ok=True)
        args.save_file = os.path.join(output_dir, filename)

        if "t2i" in args.task:
            logging.info(f"Saving generated image to {args.save_file}")
            cache_image(
                tensor=video.squeeze(1)[None],
                save_file=args.save_file,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        else:
            logging.info(f"Saving generated video to {args.save_file}")
            cache_video(
                tensor=video[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))

        ori_root = getattr(args, "ori_baseline_root", None)
        quality_baseline_path = None
        if ori_root:
            ori_root_abs = os.path.abspath(
                ori_root if os.path.isabs(ori_root) else os.path.join(_CUBIC_ROOT, ori_root)
            )
            _pid = get_prompt_id(args.prompt)
            quality_baseline_path = os.path.join(
                ori_root_abs, _pid, f"ori_{_pid}_{args.task}.mp4"
            )

        # Quality evaluation
        quality_result = evaluate_quality_with_origin(
            image_path=args.save_file,
            tag=TAG,
            save_metrics=False,
            baseline_path=quality_baseline_path,
            output_dir=output_dir,
        )
        quality_metrics = quality_result['metrics'] if quality_result is not None else None

        logging.info("Finished.")

    return quality_metrics


if __name__ == "__main__":
    os.chdir(_CUBIC_ROOT)
    args = _parse_args()
    quality_metrics = generate(args)
    print_time_statistics(timer_names=REQUIRED_TIMER_NAMES)

    rank = int(os.getenv('RANK', 0))
    if rank == 0:
        output_dir = args.output_dir if hasattr(args, 'output_dir') and args.output_dir else OUTPUT_DIR
        save_time_stats_to_file(
            os.path.join(output_dir, 'timing_stats.txt'),
            timer_names=REQUIRED_TIMER_NAMES,
        )
        params = {
            'model': 'wan',
            'method': 'pab',
            'model_id': model_id,
            'model_path': MODEL_PATH,
            'prompt': args.prompt,
            'task': args.task,
            'size': args.size,
            'base_seed': args.base_seed,
            'pab_self_range': args.pab_self_range,
            'pab_cross_range': args.pab_cross_range,
            'pab_warmup': args.pab_warmup,
            'pab_cooldown': args.pab_cooldown,
            'pab_layer_interval': args.pab_layer_interval,
            'ori_baseline_root': getattr(args, 'ori_baseline_root', None),
        }
        params_path = save_params_and_metrics(
            output_dir,
            TAG,
            params,
            get_time_statistics_dict(timer_names=REQUIRED_TIMER_NAMES),
            quality_metrics=quality_metrics,
        )
        print(f'Params & metrics saved to {params_path}', flush=True)
