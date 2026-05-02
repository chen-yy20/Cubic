#!/usr/bin/env python3
"""
预先下载 VBench 各维度测评所需权重到本地缓存目录，便于在无网络节点运行测评。
缓存路径可通过环境变量 VBENCH_CACHE_DIR 指定，默认 ~/.cache/vbench。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _download(url: str, dest_path: Path, label: str) -> None:
    if dest_path.is_file():
        print(f"[skip] {label}: {dest_path}")
        return
    _ensure_dir(dest_path.parent)
    print(f"[get] {label} <- {url}")
    urllib.request.urlretrieve(url, dest_path)


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def download_all(cache_dir: Path, clone_dino: bool) -> None:
    cache_dir = Path(cache_dir).expanduser().resolve()
    _ensure_dir(cache_dir)

    # CLIP ViT-B/32 (background_consistency, appearance_style local)
    vit_b = cache_dir / "clip_model" / "ViT-B-32.pt"
    _download(
        "https://openaipublic.azureedge.net/clip/models/"
        "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        vit_b,
        "CLIP ViT-B-32",
    )

    # CLIP ViT-L/14 (aesthetic_quality local)
    vit_l = cache_dir / "clip_model" / "ViT-L-14.pt"
    _download(
        "https://openaipublic.azureedge.net/clip/models/"
        "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
        vit_l,
        "CLIP ViT-L-14",
    )

    # UMT (human_action)
    umt = cache_dir / "umt_model" / "l16_ptk710_ftk710_ftk400_f16_res224.pth"
    _download(
        "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/"
        "l16_ptk710_ftk710_ftk400_f16_res224.pth",
        umt,
        "UMT",
    )

    # AMT (motion_smoothness)
    amt = cache_dir / "amt_model" / "amt-s.pth"
    _download(
        "https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth",
        amt,
        "AMT-S",
    )

    # RAFT (dynamic_degree)
    raft_zip = cache_dir / "raft_model" / "models.zip"
    raft_model = cache_dir / "raft_model" / "models" / "raft-things.pth"
    if not raft_model.is_file():
        _download(
            "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip",
            raft_zip,
            "RAFT zip",
        )
        _ensure_dir(raft_zip.parent)
        _run(["unzip", "-o", "-d", str(cache_dir / "raft_model"), str(raft_zip)])
        try:
            raft_zip.unlink()
        except OSError:
            pass
    else:
        print(f"[skip] RAFT: {raft_model}")

    # DINO repo + weights (subject_consistency, local mode)
    dino_repo = cache_dir / "dino_model" / "facebookresearch_dino_main"
    dino_ckpt = cache_dir / "dino_model" / "dino_vitbase16_pretrain.pth"
    if clone_dino and not dino_repo.is_dir():
        _ensure_dir(dino_repo.parent)
        _run(
            [
                "git",
                "clone",
                "https://github.com/facebookresearch/dino",
                str(dino_repo),
            ]
        )
    elif clone_dino:
        print(f"[skip] DINO repo exists: {dino_repo}")
    else:
        print(
            "[warn] 跳过 DINO git clone；若无本地仓库，测评 subject_consistency 时请设置 "
            "--clone-dino 在有网环境执行一次。"
        )

    _download(
        "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/"
        "dino_vitbase16_pretrain.pth",
        dino_ckpt,
        "DINO ViT-B/16",
    )

    # LAION aesthetic linear head（aesthetic_quality 运行时也会尝试下载，此处预先落盘）
    aes_dir = cache_dir / "aesthetic_model" / "emb_reader"
    aes_pth = aes_dir / "sa_0_4_vit_l_14_linear.pth"
    _ensure_dir(aes_dir)
    if not aes_pth.is_file():
        _download(
            "https://github.com/LAION-AI/aesthetic-predictor/raw/main/"
            "sa_0_4_vit_l_14_linear.pth",
            aes_pth,
            "LAION aesthetic linear",
        )

    # PyIQA MUSIQ (imaging_quality)
    musiq = cache_dir / "pyiqa_model" / "musiq_spaq_ckpt-358bb6af.pth"
    _download(
        "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/"
        "musiq_spaq_ckpt-358bb6af.pth",
        musiq,
        "MUSIQ",
    )

    # GRIT (object_class, multiple_objects, color, spatial_relationship)
    grit = cache_dir / "grit_model" / "grit_b_densecap_objectdet.pth"
    _download(
        "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/"
        "grit_b_densecap_objectdet.pth",
        grit,
        "GRIT",
    )

    # Tag2Text (scene)
    tag2 = cache_dir / "caption_model" / "tag2text_swin_14m.pth"
    _download(
        "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/"
        "tag2text_swin_14m.pth",
        tag2,
        "Tag2Text",
    )

    # ViCLIP (temporal_style, overall_consistency)
    viclip = cache_dir / "ViCLIP" / "ViClip-InternVid-10M-FLT.pth"
    _download(
        "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/"
        "ViClip-InternVid-10M-FLT.pth",
        viclip,
        "ViCLIP",
    )

    print(f"\n完成。请设置环境变量: export VBENCH_CACHE_DIR={cache_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 VBench 测评权重到本地缓存")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("VBENCH_CACHE_DIR")
        or str(Path.home() / ".cache" / "vbench"),
        help="权重缓存目录（与 VBENCH 的 VBENCH_CACHE_DIR 一致）",
    )
    parser.add_argument(
        "--no-clone-dino",
        action="store_true",
        help="不执行 git clone facebookresearch/dino（若已手动放置可跳过）",
    )
    args = parser.parse_args()
    try:
        download_all(Path(args.cache_dir), clone_dino=not args.no_clone_dino)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
