"""
PIE-Bench++ 批量推理共用：从 parquet 取图、将 edit_action 转为自然语言、target_prompt 展平为 hpsv3 说明。
由 pie_bench_batch_generate.py 引用。
"""

from __future__ import annotations

import io
import json
import re
from typing import Any, Dict, Union

from PIL import Image

_EDIT_TYPE_DESC: dict[int, str] = {
    1: "object change",
    2: "add object",
    3: "remove object / phrase",
    4: "attribute: content",
    5: "attribute: pose",
    6: "attribute: color",
    7: "attribute: material",
    8: "background",
    9: "style",
}


def target_prompt_to_hpsv3_caption(target_prompt: str) -> str:
    """HPSv3：去掉 PIE 方括号后的整句自然语言。"""

    return re.sub(r"\[([^\]]*)\]", r"\1", target_prompt or "").strip()


def _parse_edit_action(raw: Union[str, dict, None]) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    if not s:
        return {}
    return json.loads(s)


def edit_action_to_natural_language(
    raw: Union[str, dict, None], *, lead_in: str = "Edit the image: "
) -> str:
    """将 edit_action JSON 转为一行英文自然语言，供条件编辑模型作 prompt。"""

    data = _parse_edit_action(raw)
    if not data:
        return ""

    entries: list[tuple[int, int, str]] = []
    for idx, (span_key, info) in enumerate(data.items()):
        if not isinstance(info, dict):
            continue
        pos = int(info.get("position", 0))
        et = int(info.get("edit_type", 0))
        act = info.get("action", "")
        etn = _EDIT_TYPE_DESC.get(et, f"edit_type {et}")
        if act == "+":
            part = f"add \"{span_key}\" ({etn})"
        elif act == "-":
            part = f"remove \"{span_key}\" ({etn})"
        else:
            a = str(act) if act is not None else ""
            part = f"change \"{a}\" to \"{span_key}\" ({etn})"
        entries.append((pos, idx, part))

    entries.sort(key=lambda x: (x[0], x[1]))
    body = "; ".join(p for *_, p in entries)
    if not body:
        return ""
    return f"{lead_in}{body}."


def image_bytes_to_pil(row_image: Any) -> Image.Image:
    if row_image is None:
        raise ValueError("image column is None")
    if isinstance(row_image, dict):
        raw = row_image.get("bytes")
    else:
        raw = getattr(row_image, "bytes", None)
    if not raw:
        raise ValueError("image.bytes is empty; check parquet / download")
    return Image.open(io.BytesIO(raw)).convert("RGB")
