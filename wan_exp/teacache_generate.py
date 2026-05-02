# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Wan T2V/I2V generation with TeaCache; timing & quality aligned with cubic_generate / tocache_generate."""
import argparse
import logging
import os
import random
import sys
import warnings

warnings.filterwarnings('ignore')

# Resolve `wan` (under wan_exp) and Cubic root (`utils`, `dist`)
_WAN_EXP_ROOT = os.path.dirname(os.path.abspath(__file__))
_CUBIC_ROOT = os.path.dirname(_WAN_EXP_ROOT)
sys.path.insert(0, _WAN_EXP_ROOT)
sys.path.insert(0, _CUBIC_ROOT)

import gc
import math
from contextlib import contextmanager

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS, SUPPORTED_SIZES
from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import cache_image, cache_video, str2bool

from dist.parallel_state import init_cp_group, init_distributed_environment
from utils.quality_metric import evaluate_quality_with_origin
from utils.results import get_prompt_id, save_params_and_metrics
from utils.timer import (
    get_time_statistics_dict,
    get_timer,
    init_timer,
    print_time_statistics,
    save_time_stats_to_file,
)

init_timer()

TAG = 'teacache'
MODEL_PATH = '/home/fit/zhaijdcyy/WORK/models/Wan2.1-T2V-1.3B'
PROMPT = (
    'Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely '
    'on a spotlighted stage.'
)

model_id = '1.3B' if '1.3B' in MODEL_PATH else '14B'
OUTPUT_DIR = f'./results/wan/{model_id}/teacache/{get_prompt_id(PROMPT)}'

REQUIRED_TIMER_NAMES = [
    'generate_e2e',
    'model_forward',
]

EXAMPLE_PROMPT = {
    't2v-1.3B': {
        'prompt': PROMPT,
    },
    't2v-14B': {
        'prompt': PROMPT,
    },
    't2i-14B': {
        'prompt': '一个朴素端庄的美人',
    },
    'i2v-14B': {
        'prompt': (
            'Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. '
            'The fluffy-furred feline gazes directly at the camera with a relaxed expression. '
            'Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, '
            'and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, '
            'as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline\'s '
            'intricate details and the refreshing atmosphere of the seaside.'
        ),
        'image': 'examples/i2v_input.JPG',
    },
}

# -----------------------------------------------------------------------------
# CFG-parallel (2 GPU) — same pattern as ori_generate / cubic_generate
# -----------------------------------------------------------------------------
rank_env = int(os.getenv('RANK', 0))
world_size_env = int(os.getenv('WORLD_SIZE', 1))
local_rank_env = int(os.getenv('LOCAL_RANK', 0))
if world_size_env > 1:
    assert world_size_env == 2, 'only support cfg parallel'
    torch.cuda.set_device(local_rank_env)
    init_distributed_environment(
        world_size=world_size_env,
        rank=rank_env,
        local_rank=local_rank_env,
    )
    init_cp_group(
        group_ranks=[[0, 1]],
        local_rank=local_rank_env,
        backend='nccl',
    )


def apply_teacache_to_model(model_cls, ckpt_dir, sample_steps, teacache_thresh, use_ret_steps):
    model_cls.enable_teacache = True
    model_cls.forward = teacache_forward
    model_cls.cnt = 0
    model_cls.num_steps = sample_steps * 2
    model_cls.teacache_thresh = teacache_thresh
    model_cls.accumulated_rel_l1_distance_even = 0
    model_cls.accumulated_rel_l1_distance_odd = 0
    model_cls.previous_e0_even = None
    model_cls.previous_e0_odd = None
    model_cls.previous_residual_even = None
    model_cls.previous_residual_odd = None
    model_cls.use_ref_steps = use_ret_steps
    if use_ret_steps:
        if '1.3B' in ckpt_dir:
            model_cls.coefficients = [
                -5.21862437e04,
                9.23041404e03,
                -5.28275948e02,
                1.36987616e01,
                -4.99875664e-02,
            ]
        if '14B' in ckpt_dir:
            model_cls.coefficients = [
                -3.03318725e05,
                4.90537029e04,
                -2.65530556e03,
                5.87365115e01,
                -3.15583525e-01,
            ]
        model_cls.ret_steps = 5 * 2
        model_cls.cutoff_steps = sample_steps * 2
    else:
        if '1.3B' in ckpt_dir:
            model_cls.coefficients = [
                2.39676752e03,
                -1.31110545e03,
                2.01331979e02,
                -8.29855975e00,
                1.37887774e-01,
            ]
        if '14B' in ckpt_dir:
            model_cls.coefficients = [
                -5784.54975374,
                5449.50911966,
                -1811.16591783,
                256.27178429,
                -13.02252404,
            ]
        model_cls.ret_steps = 1 * 2
        model_cls.cutoff_steps = sample_steps * 2 - 2


def apply_teacache_i2v_coefficients(model_cls, ckpt_dir, sample_steps, use_ret_steps):
    """I2V uses different polynomial coefficients than T2V (checkpoint name hints)."""
    if use_ret_steps:
        if '480P' in ckpt_dir:
            model_cls.coefficients = [
                2.57151496e05,
                -3.54229917e04,
                1.40286849e03,
                -1.35890334e01,
                1.32517977e-01,
            ]
        elif '720P' in ckpt_dir:
            model_cls.coefficients = [
                8.10705460e03,
                2.13393892e03,
                -3.72934672e02,
                1.66203073e01,
                -4.17769401e-02,
            ]
        else:
            logging.warning(
                'TeaCache I2V: ckpt_dir does not contain 480P or 720P (use_ret_steps); path=%s',
                ckpt_dir,
            )
        model_cls.ret_steps = 5 * 2
        model_cls.cutoff_steps = sample_steps * 2
    else:
        if '480P' in ckpt_dir:
            model_cls.coefficients = [
                -3.02331670e02,
                2.23948934e02,
                -5.25463970e01,
                5.87348440e00,
                -2.01973289e-01,
            ]
        elif '720P' in ckpt_dir:
            model_cls.coefficients = [
                -114.36346466,
                65.26524496,
                -18.82220707,
                4.91518089,
                -0.23412683,
            ]
        else:
            logging.warning(
                'TeaCache I2V: ckpt_dir does not contain 480P or 720P; path=%s',
                ckpt_dir,
            )
        model_cls.ret_steps = 1 * 2
        model_cls.cutoff_steps = sample_steps * 2 - 2


def teacache_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
):
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x
    ])

    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context
        ])
    )

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
    )

    if self.enable_teacache:
        modulated_inp = e0 if self.use_ref_steps else e
        if self.cnt % 2 == 0:
            self.is_even = True
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                should_calc_even = True
                self.accumulated_rel_l1_distance_even = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                rel = (
                    (modulated_inp - self.previous_e0_even).abs().mean()
                    / self.previous_e0_even.abs().mean()
                ).cpu().item()
                self.accumulated_rel_l1_distance_even += rescale_func(rel)
                if self.accumulated_rel_l1_distance_even < self.teacache_thresh:
                    should_calc_even = False
                else:
                    should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0
            self.previous_e0_even = modulated_inp.clone()
        else:
            self.is_even = False
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                should_calc_odd = True
                self.accumulated_rel_l1_distance_odd = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                rel = (
                    (modulated_inp - self.previous_e0_odd).abs().mean()
                    / self.previous_e0_odd.abs().mean()
                ).cpu().item()
                self.accumulated_rel_l1_distance_odd += rescale_func(rel)
                if self.accumulated_rel_l1_distance_odd < self.teacache_thresh:
                    should_calc_odd = False
                else:
                    should_calc_odd = True
                    self.accumulated_rel_l1_distance_odd = 0
            self.previous_e0_odd = modulated_inp.clone()

    if self.enable_teacache:
        if self.is_even:
            if not should_calc_even:
                x = x + self.previous_residual_even
            else:
                ori_x = x.clone()
                for block in self.blocks:
                    x = block(x, **kwargs)
                self.previous_residual_even = x - ori_x
        else:
            if not should_calc_odd:
                x = x + self.previous_residual_odd
            else:
                ori_x = x.clone()
                for block in self.blocks:
                    x = block(x, **kwargs)
                self.previous_residual_odd = x - ori_x
    else:
        for block in self.blocks:
            x = block(x, **kwargs)

    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.cnt = 0
    return [u.float() for u in x]


def t2v_generate_teacache(
    self,
    input_prompt,
    size=(1280, 720),
    frame_num=81,
    shift=5.0,
    sample_solver='unipc',
    sampling_steps=50,
    guide_scale=5.0,
    n_prompt='',
    seed=-1,
    offload_model=True,
):
    F = frame_num
    target_shape = (
        self.vae.model.z_dim,
        (F - 1) // self.vae_stride[0] + 1,
        size[1] // self.vae_stride[1],
        size[0] // self.vae_stride[2],
    )

    seq_len = math.ceil(
        (target_shape[2] * target_shape[3])
        / (self.patch_size[1] * self.patch_size[2])
        * target_shape[1]
        / self.sp_size
    ) * self.sp_size

    if n_prompt == '':
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
            generator=seed_g,
        )
    ]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, 'no_sync', noop_no_sync)

    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas,
            )
        else:
            raise NotImplementedError('Unsupported solver.')

        latents = noise
        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}

        with get_timer('generate_e2e'):
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = torch.stack([t])

                self.model.to(self.device)
                with get_timer('model_forward'):
                    noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                with get_timer('model_forward'):
                    noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0]
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


def i2v_generate_teacache(
    self,
    input_prompt,
    img,
    max_area=720 * 1280,
    frame_num=81,
    shift=5.0,
    sample_solver='unipc',
    sampling_steps=40,
    guide_scale=5.0,
    n_prompt='',
    seed=-1,
    offload_model=True,
):
    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

    F = frame_num
    h, w = img.shape[1:]
    aspect_ratio = h / w
    lat_h = round(
        np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1]
    )
    lat_w = round(
        np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2]
    )
    h = lat_h * self.vae_stride[1]
    w = lat_w * self.vae_stride[2]

    max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
        self.patch_size[1] * self.patch_size[2]
    )
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
        device=self.device,
    )

    msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
    msk[:, 1:] = 0
    msk = torch.concat(
        [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
        dim=1,
    )
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    if n_prompt == '':
        n_prompt = self.sample_neg_prompt

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

    y = self.vae.encode(
        [
            torch.concat(
                [
                    torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
                    torch.zeros(3, F - 1, h, w),
                ],
                dim=1,
            ).to(self.device)
        ]
    )[0]
    y = torch.concat([msk, y])

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, 'no_sync', noop_no_sync)

    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas,
            )
        else:
            raise NotImplementedError('Unsupported solver.')

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
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = torch.stack([t]).to(self.device)

                with get_timer('model_forward'):
                    noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0].to(
                        torch.device('cpu') if offload_model else self.device
                    )
                if offload_model:
                    torch.cuda.empty_cache()

                with get_timer('model_forward'):
                    noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0].to(
                        torch.device('cpu') if offload_model else self.device
                    )
                if offload_model:
                    torch.cuda.empty_cache()

                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                latent = latent.to(torch.device('cpu') if offload_model else self.device)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0]
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
    assert args.ckpt_dir is not None, 'Please specify the checkpoint directory.'
    assert args.task in WAN_CONFIGS, f'Unsupport task: {args.task}'
    assert args.task in EXAMPLE_PROMPT, f'Unsupport task: {args.task}'

    if args.sample_steps is None:
        args.sample_steps = 40 if 'i2v' in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if 'i2v' in args.task and args.size in ['832*480', '480*832']:
            args.sample_shift = 3.0

    if args.frame_num is None:
        args.frame_num = 1 if 't2i' in args.task else 81

    if 't2i' in args.task:
        assert args.frame_num == 1, f'Unsupport frame_num {args.frame_num} for task {args.task}'

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)

    assert args.size in SUPPORTED_SIZES[args.task], (
        f'Unsupport size {args.size} for task {args.task}, '
        f"supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Generate image or video with Wan + TeaCache (timing & metrics aligned with cubic_generate).'
    )
    parser.add_argument('--task', type=str, default='t2v-14B', choices=list(WAN_CONFIGS.keys()))
    parser.add_argument(
        '--size',
        type=str,
        default='1280*720',
        choices=list(SIZE_CONFIGS.keys()),
    )
    parser.add_argument('--frame_num', type=int, default=None)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--offload_model', type=str2bool, default=None)
    parser.add_argument('--ulysses_size', type=int, default=1)
    parser.add_argument('--ring_size', type=int, default=1)
    parser.add_argument('--t5_fsdp', action='store_true', default=False)
    parser.add_argument('--t5_cpu', action='store_true', default=False)
    parser.add_argument('--dit_fsdp', action='store_true', default=False)
    parser.add_argument('--save_file', type=str, default=None)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--use_prompt_extend', action='store_true', default=False)
    parser.add_argument(
        '--prompt_extend_method',
        type=str,
        default='local_qwen',
        choices=['dashscope', 'local_qwen'],
    )
    parser.add_argument('--prompt_extend_model', type=str, default=None)
    parser.add_argument('--prompt_extend_target_lang', type=str, default='ch', choices=['ch', 'en'])
    parser.add_argument('--base_seed', type=int, default=-1)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--sample_solver', type=str, default='unipc', choices=['unipc', 'dpm++'])
    parser.add_argument('--sample_steps', type=int, default=None)
    parser.add_argument('--sample_shift', type=float, default=None)
    parser.add_argument('--sample_guide_scale', type=float, default=5.0)
    parser.add_argument('--teacache_thresh', type=float, default=0.07)
    parser.add_argument('--use_ret_steps', action='store_true', default=False)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument(
        '--ori_baseline_root',
        type=str,
        default=None,
        help=(
            'Ori 结果根目录（含各 prompt_id 子目录）。设置后与对应 prompt 的 ori 视频比对质量；'
            '用于输出目录名为 teacache_t0pXX 等无法自动推断的情形。默认 None 沿用原推断逻辑。'
        ),
    )

    args = parser.parse_args()

    args.task = 't2v-1.3B' if '1.3B' in MODEL_PATH else 't2v-14B'
    args.size = '832*480' if '1.3B' in MODEL_PATH else '1280*720'
    if args.prompt is None:
        args.prompt = PROMPT
    args.ckpt_dir = MODEL_PATH
    args.base_seed = 42 if args.base_seed < 0 else args.base_seed

    _validate_args(args)
    return args


def _init_logging(cur_rank):
    if cur_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    device = local_rank
    _init_logging(rank)

    output_dir = getattr(args, 'output_dir', None) or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f'offload_model is not specified, set to {args.offload_model}.')

    if world_size == 1:
        assert not (args.t5_fsdp or args.dit_fsdp), (
            't5_fsdp and dit_fsdp are not supported in non-distributed environments.'
        )
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), 'context parallel are not supported in non-distributed environments.'
    elif args.ulysses_size > 1 or args.ring_size > 1:
        raise NotImplementedError('ulysses/ring parallel is not supported with cfg group init mode')

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, '`num_heads` must be divisible by `ulysses_size`.'

    logging.info(f'Generation job args: {args}')
    logging.info(f'Generation model config: {cfg}')

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    quality_metrics = None

    if 't2v' in args.task or 't2i' in args.task:
        logging.info(f'Input prompt: {args.prompt}')

        logging.info('Creating WanT2V pipeline with TeaCache.')
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

        wan_t2v.__class__.generate = t2v_generate_teacache
        apply_teacache_to_model(
            wan_t2v.model.__class__,
            args.ckpt_dir,
            args.sample_steps,
            args.teacache_thresh,
            args.use_ret_steps,
        )

        logging.info(f"Generating {'image' if 't2i' in args.task else 'video'} ...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
        )

    else:
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]['image']
        logging.info(f'Input prompt: {args.prompt}')
        logging.info(f'Input image: {args.image}')

        img = Image.open(args.image).convert('RGB')

        logging.info('Creating WanI2V pipeline with TeaCache.')
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

        wan_i2v.__class__.generate = i2v_generate_teacache
        apply_teacache_to_model(
            wan_i2v.model.__class__,
            args.ckpt_dir,
            args.sample_steps,
            args.teacache_thresh,
            args.use_ret_steps,
        )
        apply_teacache_i2v_coefficients(
            wan_i2v.model.__class__,
            args.ckpt_dir,
            args.sample_steps,
            args.use_ret_steps,
        )

        logging.info('Generating video ...')
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
            offload_model=args.offload_model,
        )

    if rank == 0:
        prompt_for_name = args.prompt
        filename = (
            f'{TAG}_{get_prompt_id(prompt_for_name)}_{args.task}.mp4'
            if 't2i' not in args.task
            else f'{TAG}_{get_prompt_id(prompt_for_name)}_{args.task}.png'
        )
        os.makedirs(output_dir, exist_ok=True)
        args.save_file = os.path.join(output_dir, filename)

        ori_root = getattr(args, 'ori_baseline_root', None)
        quality_baseline_path = None
        if ori_root:
            ori_root_abs = os.path.abspath(
                ori_root if os.path.isabs(ori_root) else os.path.join(_CUBIC_ROOT, ori_root)
            )
            _pid = get_prompt_id(prompt_for_name)
            quality_baseline_path = os.path.join(
                ori_root_abs, _pid, f'ori_{_pid}_{args.task}.mp4'
            )

        if 't2i' in args.task:
            logging.info(f'Saving generated image to {args.save_file}')
            cache_image(
                tensor=video.squeeze(1)[None],
                save_file=args.save_file,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
        else:
            logging.info(f'Saving generated video to {args.save_file}')
            cache_video(
                tensor=video[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )

        quality_result = evaluate_quality_with_origin(
            image_path=args.save_file,
            tag=TAG,
            save_metrics=False,
            baseline_path=quality_baseline_path,
            output_dir=output_dir,
        )
        quality_metrics = quality_result['metrics'] if quality_result is not None else None

        logging.info('Finished.')
        args.output_dir = output_dir
        return quality_metrics

    logging.info('Finished.')
    return None


if __name__ == '__main__':
    os.chdir(_CUBIC_ROOT)

    args = _parse_args()
    quality_metrics = generate(args)

    print_time_statistics(timer_names=REQUIRED_TIMER_NAMES)

    rank = int(os.getenv('RANK', 0))
    if rank == 0:
        out_dir = getattr(args, 'output_dir', None) or OUTPUT_DIR
        save_time_stats_to_file(
            os.path.join(out_dir, 'timing_stats.txt'),
            timer_names=REQUIRED_TIMER_NAMES,
        )
        params = {
            'model': 'wan',
            'method': 'teacache',
            'model_id': model_id,
            'model_path': MODEL_PATH,
            'prompt': args.prompt,
            'task': args.task,
            'size': args.size,
            'base_seed': args.base_seed,
            'teacache_thresh': args.teacache_thresh,
            'use_ret_steps': args.use_ret_steps,
            'ori_baseline_root': getattr(args, 'ori_baseline_root', None),
        }
        params_path = save_params_and_metrics(
            out_dir,
            TAG,
            params,
            get_time_statistics_dict(timer_names=REQUIRED_TIMER_NAMES),
            quality_metrics=quality_metrics,
        )
        print(f'Params & metrics saved to {params_path}', flush=True)
