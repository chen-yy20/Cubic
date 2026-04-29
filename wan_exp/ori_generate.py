# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import cache_video, str2bool

from utils.timer import init_timer, print_time_statistics, get_time_statistics_dict
from utils.results import save_params_and_metrics, get_prompt_id
from dist.parallel_state import (
    init_distributed_environment,
    init_cp_group,
)

init_timer()

PROMPT = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
MODEL_PATH = "/home/fit/zhaijdcyy/WORK/models/Wan2.1-T2V-1.3B"

TAG = "ori"
model_id = "1.3B" if "1.3B" in MODEL_PATH else "14B"
OUTPUT_DIR = f"./results/wan/{model_id}/ori/{get_prompt_id(PROMPT)}"

# 2卡并行初始化（CFG parallel）
rank = int(os.getenv("RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
local_rank = int(os.getenv("LOCAL_RANK", 0))
if world_size > 1:
    assert world_size == 2, "only support cfg parallel"
    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
    )
    init_cp_group(
        group_ranks=[[0, 1]],
        local_rank=local_rank,
        backend="nccl",
    )


def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in ["t2v-1.3B", "t2v-14B"], f"Unsupport task: {args.task}"
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.size in SUPPORTED_SIZES[args.task], (
        f"Unsupport size {args.size} for task {args.task}, "
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
        description="Generate a video from text prompt using WanT2V"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=["t2v-1.3B", "t2v-14B"],
        help="The task to run.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video.",
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a video. The number should be 4n+1",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.",
    )
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward.",
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.",
    )
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.",
    )
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.",
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.",
    )
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.",
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.",
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default="unipc",
        choices=["unipc", "dpm++"],
        help="The solver used to sample.",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=None,
        help="The sampling steps.",
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.",
    )
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/wan",
        help="output_dir.",
    )

    args = parser.parse_args()
    args.task = "t2v-1.3B" if "1.3B" in MODEL_PATH else "t2v-14B"
    args.size = "832*480" if "1.3B" in MODEL_PATH else "1280*720"
    args.prompt = PROMPT
    args.ckpt_dir = MODEL_PATH
    args.output_dir = OUTPUT_DIR
    args.base_seed = 42 if args.base_seed < 0 else args.base_seed

    _validate_args(args)
    return args


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")

    if world_size == 1:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), "context parallel are not supported in non-distributed environments."
    elif args.ulysses_size > 1 or args.ring_size > 1:
        raise NotImplementedError("ulysses/ring parallel is not supported with cfg group init mode")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, "`num_heads` must be divisible by `ulysses_size`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info("Creating WanT2V pipeline.")
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        t5_cpu=args.t5_cpu,
    )

    logging.info("Generating video ...")
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

    if rank == 0:
        filename = f"{TAG}_{get_prompt_id(PROMPT)}_{args.task}.mp4"
        os.makedirs(args.output_dir, exist_ok=True)
        args.save_file = os.path.join(args.output_dir, filename)

        logging.info(f"Saving generated video to {args.save_file}")
        cache_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
    print_time_statistics()

    rank = int(os.getenv("RANK", 0))
    if rank == 0:
        params = {
            "model": "wan",
            "method": "ori",
            "model_id": model_id,
            "model_path": MODEL_PATH,
            "prompt": PROMPT,
            "task": args.task,
            "size": args.size,
            "base_seed": args.base_seed,
        }
        params_path = save_params_and_metrics(
            OUTPUT_DIR,
            TAG,
            params,
            get_time_statistics_dict(),
            quality_metrics=None,
        )
        print(f"Params & metrics saved to {params_path}", flush=True)
