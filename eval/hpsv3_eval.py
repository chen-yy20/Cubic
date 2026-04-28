import argparse
import os
import time

from hpsv3 import HPSv3RewardInferencer


def _log(msg: str) -> None:
    print(f"[hpsv3_eval] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="HPSv3 reward eval (offline-friendly)")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("HPSV3_CHECKPOINT"),
        help="本地 HPSv3.safetensors 路径；不设则走 Hugging Face（计算节点需能联网或已有缓存）",
    )
    args = parser.parse_args()

    kwargs: dict = {"device": "cuda"}
    if args.checkpoint:
        kwargs["checkpoint_path"] = args.checkpoint

    _log("开始构建 HPSv3RewardInferencer（读权重 + 建模型 + 上 GPU，大模型可能要数分钟，尤其权重在 NFS 上）…")
    t0 = time.perf_counter()
    inferencer = HPSv3RewardInferencer(**kwargs)
    _log(f"Inferencer 就绪，耗时 {time.perf_counter() - t0:.1f}s")
    image_paths = [
        "/home/fit/zhaijdcyy/WORK/Cubic/results/flux/cubic/photorealistic_cute_cat,_wearing_4x/cubic_photorealistic_cute_cat,_wearing_seed43.png",
        "/home/fit/zhaijdcyy/WORK/Cubic/results/flux/ori/photorealistic_cute_cat,_wearing/ori_photorealistic_cute_cat,_wearing_seed43.png",
    ]
    prompts = [
        "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background.",
        "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background.",
    ]

    _log("开始 reward 推理…")
    t1 = time.perf_counter()
    rewards = inferencer.reward(image_paths, prompts)
    _log(f"推理结束，耗时 {time.perf_counter() - t1:.1f}s")
    scores = [reward[0].item() for reward in rewards]
    print(f"Image scores: {scores}")


if __name__ == "__main__":
    main()
