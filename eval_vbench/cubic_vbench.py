"""
扩展官方 VBench：标准模式下不再用「目录里第一个文件」的后缀推断视频格式，
避免 metadata.jsonl / txt 与 mp4 混放时误用 .jsonl。

默认按 .mp4 查找 `{prompt}-0.mp4` … `{prompt}-4.mp4`。

evaluate：可对「当前 meta 下没有任何视频」的维度自动跳过，避免官方 subject_consistency 除零崩溃。
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

from vbench import VBench
from vbench.distributed import get_rank, print0
from vbench.utils import init_submodules, load_dimension_info, load_json, save_json


def resolve_video_postfix(video_names: list[str], video_extension: str | None) -> str:
    if video_extension:
        ext = video_extension.strip()
        return ext if ext.startswith(".") else f".{ext}"
    for ext in (".mp4", ".MP4", ".gif", ".GIF"):
        if any(name.endswith(ext) for name in video_names):
            return ext
    return ".mp4"


class CubicVBench(VBench):
    def build_full_info_json(
        self,
        videos_path,
        name,
        dimension_list,
        prompt_list=None,
        special_str="",
        verbose=False,
        custom_image_folder=None,
        mode="vbench_standard",
        video_extension=None,
        **kwargs,
    ):
        if mode != "vbench_standard":
            return super().build_full_info_json(
                videos_path,
                name,
                dimension_list,
                prompt_list if prompt_list is not None else [],
                special_str=special_str,
                verbose=verbose,
                custom_image_folder=custom_image_folder,
                mode=mode,
                **kwargs,
            )

        full_info_list = load_json(self.full_info_dir)
        video_names = os.listdir(videos_path)
        postfix = resolve_video_postfix(video_names, video_extension)

        cur_full_info_list = []
        for prompt_dict in full_info_list:
            if set(dimension_list) & set(prompt_dict["dimension"]):
                prompt = prompt_dict["prompt_en"]
                prompt_dict["video_list"] = []
                for i in range(5):
                    intended_video_name = f"{prompt}{special_str}-{str(i)}{postfix}"
                    if intended_video_name in video_names:
                        intended_video_path = os.path.join(videos_path, intended_video_name)
                        prompt_dict["video_list"].append(intended_video_path)
                        if verbose:
                            print0(f"Successfully found video: {intended_video_name}")
                    else:
                        print0(
                            "WARNING!!! This required video is not found! Missing benchmark videos "
                            f"can lead to unfair evaluation result. The missing video is: {intended_video_name}"
                        )
                cur_full_info_list.append(prompt_dict)

        total_paths = sum(len(d.get("video_list", [])) for d in cur_full_info_list)
        if total_paths == 0:
            raise RuntimeError(
                "在 videos_path 下没有找到任何符合 VBench 命名的视频文件。\n"
                f"  目录: {videos_path}\n"
                f"  当前使用的扩展名: {postfix}\n"
                "请确认：\n"
                "  1) --videos-path 指向「仅含生成视频」的文件夹（或至少包含 mp4），而不是只有 metadata.jsonl / metrics 的汇总目录；\n"
                "  2) 文件命名为 `{prompt_en}-0.mp4` … `{prompt_en}-4.mp4`，与 VBench JSON 中 prompt 完全一致；\n"
                "  3) 若视频为 .gif，请传入 --video-extension .gif。"
            )

        cur_full_info_path = os.path.join(self.output_path, name + "_full_info.json")
        save_json(cur_full_info_list, cur_full_info_path)
        print0(f"Evaluation meta data saved to {cur_full_info_path}")
        return cur_full_info_path

    def evaluate(
        self,
        videos_path,
        name,
        prompt_list=None,
        dimension_list=None,
        local=False,
        read_frame=False,
        mode="vbench_standard",
        skip_empty_dimensions=True,
        **kwargs,
    ):
        """与官方一致；skip_empty_dimensions=True 时跳过 meta 中无视频的维度，避免除零。"""
        if prompt_list is None:
            prompt_list = []
        kw = dict(kwargs)
        if "skip_empty_dimensions" in kw:
            skip_empty_dimensions = kw.pop("skip_empty_dimensions")

        results_dict = {}
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()
        submodules_dict = init_submodules(
            dimension_list, local=local, read_frame=read_frame
        )

        cur_full_info_path = self.build_full_info_json(
            videos_path,
            name,
            dimension_list,
            prompt_list,
            mode=mode,
            **kw,
        )

        kw_compute = dict(kw)
        kw_compute.pop("video_extension", None)

        for dimension in dimension_list:
            if skip_empty_dimensions:
                video_list, _ = load_dimension_info(
                    cur_full_info_path, dimension, "en"
                )
                if not video_list:
                    print0(
                        f"[CubicVBench] 跳过维度 {dimension}：meta 中该维度无任何视频路径。"
                    )
                    results_dict[dimension] = {"skipped_no_videos": True}
                    continue
            try:
                dimension_module = importlib.import_module(f"vbench.{dimension}")
                evaluate_func = getattr(
                    dimension_module, f"compute_{dimension}"
                )
            except Exception as e:
                raise NotImplementedError(
                    f"UnImplemented dimension {dimension}!, {e}"
                ) from e
            submodules_list = submodules_dict[dimension]
            print0(f"cur_full_info_path: {cur_full_info_path}")
            results = evaluate_func(
                cur_full_info_path, self.device, submodules_list, **kw_compute
            )
            results_dict[dimension] = results
        output_name = os.path.join(self.output_path, name + "_eval_results.json")
        if get_rank() == 0:
            save_json(results_dict, output_name)
            print0(f"Evaluation results saved to {output_name}")
