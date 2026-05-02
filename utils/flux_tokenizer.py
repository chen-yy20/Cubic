"""Flux ``from_pretrained`` 辅助：供 pre_exp 脚本统一加载参数。"""

from __future__ import annotations

import os
from typing import Any


def flux_from_pretrained_kwargs(model_path: str, dtype: Any) -> dict[str, Any]:
    """返回 ``FluxPipeline.from_pretrained(model_path, **kwargs)`` 的 kwargs。

    与 ``flux1_exp/cubic_generate.py`` 中 ``CubicFluxPipeline.from_pretrained``
    的用法一致；若设置了 ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``，则传给 Hub（ gated 模型）。
    """
    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        kwargs["token"] = token
    path = os.path.expanduser(model_path)
    if os.path.isdir(path):
        kwargs["local_files_only"] = True
    return kwargs
