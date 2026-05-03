# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Wan T2V with ToCa baseline (`wan.jano_baselines.text2video_tocache`); timing & quality aligned with cubic_generate."""
import argparse
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

_WAN_EXP_ROOT = os.path.dirname(os.path.abspath(__file__))
_CUBIC_ROOT = os.path.dirname(_WAN_EXP_ROOT)
sys.path.insert(0, _WAN_EXP_ROOT)
sys.path.insert(0, _CUBIC_ROOT)

import torch
import torch.distributed as dist

from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
from wan.jano_baselines.text2video_tocache import WanT2V_toca
from wan.utils.utils import cache_video, str2bool

from dist.parallel_state import (
    init_cp_group,
    init_distributed_environment,
)
from utils.launch_defaults import resolve_model_path
from utils.quality_metric import evaluate_quality_with_origin
from utils.results import get_prompt_id, save_params_and_metrics
from utils.timer import (
    get_time_statistics_dict,
    init_timer,
    print_time_statistics,
    save_time_stats_to_file,
)

init_timer()

TAG = 'tocache'
PROMPT = (
    'Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely '
    'on a spotlighted stage.'
)

# Baseline `WanT2V_toca.generate` wraps the diffusion loop in `get_timer("generate_e2e")`.
REQUIRED_TIMER_NAMES = [
    'generate_e2e',
]


def _resolve_model_defaults(model_path):
    if '1.3B' in model_path:
        return 't2v-1.3B', '832*480', '1.3B'
    return 't2v-14B', '1280*720', '14B'


def _default_output_dir(model_path, prompt):
    _, _, model_id = _resolve_model_defaults(model_path)
    return f'./results/wan/{model_id}/tocache/{get_prompt_id(prompt)}'

rank = int(os.getenv('RANK', 0))
world_size = int(os.getenv('WORLD_SIZE', 1))
local_rank = int(os.getenv('LOCAL_RANK', 0))
if world_size > 1:
    assert world_size == 2, 'only support cfg parallel'
    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
    )
    init_cp_group(
        group_ranks=[[0, 1]],
        local_rank=local_rank,
        backend='nccl',
    )


def _validate_args(args):
    assert args.ckpt_dir is not None, 'Please specify the checkpoint directory.'
    assert args.task in ['t2v-1.3B', 't2v-14B'], f'Unsupport task: {args.task}'
    assert args.task in WAN_CONFIGS, f'Unsupport task: {args.task}'
    assert args.size in SUPPORTED_SIZES[args.task], (
        f'Unsupport size {args.size} for task {args.task}, '
        f"supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"
    )

    if args.sample_steps is None:
        args.sample_steps = 50

    if args.sample_shift is None:
        args.sample_shift = 5.0

    if args.frame_num is None:
        args.frame_num = 81


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Generate a video from text prompt using Wan ToCa baseline (WanT2V_toca)'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='t2v-14B',
        choices=['t2v-1.3B', 't2v-14B'],
        help='The task to run.',
    )
    parser.add_argument(
        '--size',
        type=str,
        default='1280*720',
        choices=list(SIZE_CONFIGS.keys()),
        help='The area (width*height) of the generated video.',
    )
    parser.add_argument(
        '--frame_num',
        type=int,
        default=None,
        help='How many frames to sample from a video. The number should be 4n+1',
    )
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default=None,
        help='The path to the checkpoint directory.',
    )
    parser.add_argument(
        '--offload_model',
        type=str2bool,
        default=None,
        help='Whether to offload the model to CPU after each model forward.',
    )
    parser.add_argument(
        '--ulysses_size',
        type=int,
        default=1,
        help='The size of the ulysses parallelism in DiT.',
    )
    parser.add_argument(
        '--ring_size',
        type=int,
        default=1,
        help='The size of the ring attention parallelism in DiT.',
    )
    parser.add_argument(
        '--t5_fsdp',
        action='store_true',
        default=False,
        help='Whether to use FSDP for T5.',
    )
    parser.add_argument(
        '--t5_cpu',
        action='store_true',
        default=False,
        help='Whether to place T5 model on CPU.',
    )
    parser.add_argument(
        '--dit_fsdp',
        action='store_true',
        default=False,
        help='Whether to use FSDP for DiT.',
    )
    parser.add_argument(
        '--save_file',
        type=str,
        default=None,
        help='The file to save the generated video to.',
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='The prompt to generate the video from.',
    )
    parser.add_argument(
        '--base_seed',
        type=int,
        default=-1,
        help='The seed to use for generating the video.',
    )
    parser.add_argument(
        '--sample_solver',
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help='The solver used to sample.',
    )
    parser.add_argument(
        '--sample_steps',
        type=int,
        default=None,
        help='The sampling steps.',
    )
    parser.add_argument(
        '--sample_shift',
        type=float,
        default=None,
        help='Sampling shift factor for flow matching schedulers.',
    )
    parser.add_argument(
        '--sample_guide_scale',
        type=float,
        default=5.0,
        help='Classifier free guidance scale.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='output_dir.',
    )

    args = parser.parse_args()
    args.ckpt_dir = resolve_model_path(args.ckpt_dir, '--ckpt_dir')
    args.task, args.size, _ = _resolve_model_defaults(args.ckpt_dir)
    if args.prompt is None:
        args.prompt = PROMPT
    if args.output_dir is None:
        args.output_dir = _default_output_dir(args.ckpt_dir, args.prompt)
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
    cur_rank = int(os.getenv('RANK', 0))
    cur_world_size = int(os.getenv('WORLD_SIZE', 1))
    cur_local_rank = int(os.getenv('LOCAL_RANK', 0))
    device = cur_local_rank
    _init_logging(cur_rank)

    output_dir = args.output_dir
    if cur_rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    if args.offload_model is None:
        args.offload_model = False if cur_world_size > 1 else True
        logging.info('offload_model is not specified, set to %s.', args.offload_model)

    if cur_world_size == 1:
        assert not (args.t5_fsdp or args.dit_fsdp), (
            't5_fsdp and dit_fsdp are not supported in non-distributed environments.'
        )
        assert not (args.ulysses_size > 1 or args.ring_size > 1), (
            'context parallel are not supported in non-distributed environments.'
        )
    elif args.ulysses_size > 1 or args.ring_size > 1:
        raise NotImplementedError('ulysses/ring parallel is not supported with cfg group init mode')

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, (
            '`num_heads` must be divisible by `ulysses_size`.'
        )

    logging.info('Generation job args: %s', args)
    logging.info('Generation model config: %s', cfg)

    if dist.is_initialized():
        base_seed = [args.base_seed] if cur_rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info('Creating WanT2V_toca (ToCa baseline from wan/jano_baselines).')
    wan_t2v = WanT2V_toca(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=cur_rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )

    logging.info('Generating video ...')
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

    if cur_rank == 0:
        filename = f'{TAG}_{get_prompt_id(args.prompt)}_{args.task}.mp4'
        os.makedirs(output_dir, exist_ok=True)
        args.save_file = os.path.join(output_dir, filename)

        logging.info('Saving generated video to %s', args.save_file)
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
            baseline_path=None,
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

    cur_rank = int(os.getenv('RANK', 0))
    if cur_rank == 0:
        out_dir = args.output_dir
        save_time_stats_to_file(
            os.path.join(out_dir, 'timing_stats.txt'),
            timer_names=REQUIRED_TIMER_NAMES,
        )
        _, _, model_id = _resolve_model_defaults(args.ckpt_dir)
        params = {
            'model': 'wan',
            'method': 'tocache',
            'model_id': model_id,
            'model_path': args.ckpt_dir,
            'prompt': args.prompt,
            'task': args.task,
            'size': args.size,
            'base_seed': args.base_seed,
        }
        params_path = save_params_and_metrics(
            out_dir,
            TAG,
            params,
            get_time_statistics_dict(timer_names=REQUIRED_TIMER_NAMES),
            quality_metrics=quality_metrics,
        )
        print(f'Params & metrics saved to {params_path}', flush=True)
