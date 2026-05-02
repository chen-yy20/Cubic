"""
TaylorSeer 风格缓存推理（适配 FireRed `QwenImageTransformer2DModel`）。

参考 TaylorSeer-Wan2.1：
- `cache_functions/cal_type.py`、`force_scheduler.py`、`cache_init.py` 的调度逻辑；
- `taylorseer_utils` 的 `derivative_approximation` / `taylor_formula`。

FireRed 为 Qwen 双路 DiT，未在子模块级接入 Wan 的 block 内 cache_step；此处在
**整块 transformer_blocks 前后** 对 encoder / image 两路残差分别做 Taylor 系数估计与
外推，CFG 两路对应 Wan 的 cond_stream / uncond_stream，仅在首路（branch_0）上调用
`cal_type` 以与 Wan 一致。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Union

import numpy as np
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import apply_lora_scale, deprecate, logging

from firered.transformer import compute_text_seq_len_from_mask

logger = logging.get_logger(__name__)

LAYER_IDX = 0
MOD_ENC = "encoder_residual"
MOD_HID = "hidden_residual"


def _force_scheduler(cache_dic: Dict, current: Dict) -> None:
    if cache_dic.get("fresh_ratio", 0.0) == 0.0:
        linear_step_weight = 0.0
    else:
        linear_step_weight = 0.0
    step_factor = torch.tensor(
        1 - linear_step_weight + 2 * linear_step_weight * current["step"] / max(current["num_steps"], 1)
    )
    threshold = torch.round(cache_dic["fresh_threshold"] / step_factor)
    cache_dic["cal_threshold"] = int(threshold.item())


def _cal_type(cache_dic: Dict, current: Dict) -> None:
    """与 Wan `cal_type` 一致，用于决定本步为 full 还是 Taylor。"""
    if (cache_dic.get("fresh_ratio", 0.0) == 0.0) and (not cache_dic.get("taylor_cache", False)):
        first_step = current["step"] == 0
    else:
        first_step = current["step"] < cache_dic.get("first_enhance", 1)

    force_fresh = cache_dic.get("force_fresh", "global")
    if not first_step:
        fresh_interval = cache_dic.get("cal_threshold", cache_dic["fresh_threshold"])
    else:
        fresh_interval = cache_dic["fresh_threshold"]

    if first_step or (cache_dic["cache_counter"] == fresh_interval - 1):
        current["type"] = "full"
        cache_dic["cache_counter"] = 0
        current["activated_steps"].append(current["step"])
        _force_scheduler(cache_dic, current)
    elif cache_dic.get("taylor_cache", False):
        cache_dic["cache_counter"] += 1
        current["type"] = "Taylor"
    elif cache_dic["cache_counter"] % 2 == 1:
        cache_dic["cache_counter"] += 1
        current["type"] = "ToCa"
    elif cache_dic.get("Delta-DiT", False):
        cache_dic["cache_counter"] += 1
        current["type"] = "Delta-Cache"
    else:
        cache_dic["cache_counter"] += 1
        current["type"] = "ToCa"
    if force_fresh != "global":
        pass


def cache_init_firered(num_steps: int = 50) -> tuple[dict, dict]:
    """Taylor 模式初始化（对齐 Wan `cache_init` 中 mode=='Taylor' 的配置）。"""
    cache_dic: Dict[str, Any] = {}
    cache: Dict[int, Any] = {}
    cache[-1] = {"cond_stream": {LAYER_IDX: {}}, "uncond_stream": {LAYER_IDX: {}}}
    cache_dic["cache_counter"] = 0
    cache_dic["cache"] = cache
    cache_dic["fresh_threshold"] = 2
    cache_dic["taylor_cache"] = True
    cache_dic["max_order"] = 1
    cache_dic["first_enhance"] = 1
    cache_dic["fresh_ratio"] = 0.0
    cache_dic["Delta-DiT"] = False
    cache_dic["force_fresh"] = "global"
    cache_dic["cache_type"] = "random"
    cache_dic["fresh_ratio_schedule"] = "ToCa"

    current: Dict[str, Any] = {
        "activated_steps": [],
        "step": 0,
        "num_steps": num_steps,
        "layer": LAYER_IDX,
    }
    return cache_dic, current


def _derivative_approximation(
    cache_dic: Dict,
    current: Dict,
    stream: str,
    module_key: str,
    feature: torch.Tensor,
) -> None:
    acts = current["activated_steps"]
    if len(acts) < 2:
        difference_distance = 1
    else:
        difference_distance = max(int(acts[-1]) - int(acts[-2]), 1)

    updated: Dict[int, torch.Tensor] = {}
    updated[0] = feature
    layer_cache = cache_dic["cache"][-1][stream][LAYER_IDX]
    prev = layer_cache.get(module_key, {})

    for i in range(int(cache_dic["max_order"])):
        if prev.get(i) is not None and (current["step"] > cache_dic["first_enhance"] - 2):
            updated[i + 1] = (updated[i] - prev[i]) / float(difference_distance)
        else:
            break
    layer_cache[module_key] = updated


def _taylor_formula(derivative_dict: Dict[int, torch.Tensor], distance: int) -> torch.Tensor:
    out = torch.zeros_like(derivative_dict[0])
    for i in range(len(derivative_dict)):
        out = out + (1.0 / math.factorial(i)) * derivative_dict[i] * (float(distance) ** i)
    return out


def _reset_taylorseer_runtime(self) -> None:
    n = int(getattr(self, "num_steps", 50))
    self.taylorseer_cache_dic, self.taylorseer_current = cache_init_firered(n)
    self.taylorseer_runtime = {
        "last_timestep": None,
        "step_index": 0,
        "call_index_for_step": -1,
    }
    self._firered_taylor_forward_type: str | None = None


def _ensure_taylorseer_runtime(self) -> None:
    if not hasattr(self, "taylorseer_cache_dic"):
        _reset_taylorseer_runtime(self)


def reset_taylorseer_state(self) -> None:
    """新一次生成前清空 TaylorSeer 状态（含 TeaCache 式 CFG 步进）。"""
    _reset_taylorseer_runtime(self)


def _update_cfg_runtime(self, timestep: torch.Tensor, num_steps: int) -> tuple[int, str]:
    """与 `teacache_firered._update_teacache_runtime` 相同：区分 CFG 两路调用次序。"""
    _ensure_taylorseer_runtime(self)
    current_ts = float(timestep.detach().flatten()[0].float().cpu().item())
    rt = self.taylorseer_runtime
    last = rt["last_timestep"]
    same = last is not None and abs(current_ts - last) <= 1e-12
    if last is None:
        rt["step_index"] = 0
        rt["call_index_for_step"] = 0
    elif same:
        rt["call_index_for_step"] += 1
    else:
        new_gen = current_ts > last or rt["step_index"] >= max(int(num_steps) - 1, 0)
        if new_gen:
            _reset_taylorseer_runtime(self)
            rt = self.taylorseer_runtime
            rt["step_index"] = 0
            rt["call_index_for_step"] = 0
        else:
            rt["step_index"] += 1
            rt["call_index_for_step"] = 0
    rt["last_timestep"] = current_ts
    branch = f"branch_{min(rt['call_index_for_step'], 1)}"
    return int(rt["step_index"]), branch


@apply_lora_scale("attention_kwargs")
def taylorseer_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes: list[tuple[int, int, int]] | None = None,
    txt_seq_lens: list[int] | None = None,
    guidance: torch.Tensor = None,
    attention_kwargs: dict[str, Any] | None = None,
    controlnet_block_samples=None,
    additional_t_cond=None,
    image_token_indices: torch.Tensor | None = None,
    return_dict: bool = True,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    if txt_seq_lens is not None:
        deprecate(
            "txt_seq_lens",
            "0.39.0",
            "Passing `txt_seq_lens` is deprecated and will be removed in version 0.39.0. "
            "Please use `encoder_hidden_states_mask` instead.",
            standard_warn=False,
        )

    enable = bool(getattr(self, "enable_taylorseer", False))
    num_steps = int(getattr(self, "num_steps", 50))
    cache_device = getattr(self, "taylorseer_cache_device", hidden_states.device)

    step_index, branch_key = _update_cfg_runtime(self, timestep, num_steps)
    stream = "cond_stream" if branch_key == "branch_0" else "uncond_stream"

    hidden_states = self.img_in(hidden_states)

    if image_token_indices is not None and hasattr(self, "_normalize_image_token_indices"):
        image_token_indices = self._normalize_image_token_indices(
            image_token_indices=image_token_indices,
            expected_length=hidden_states.shape[1],
            device=hidden_states.device,
        )

    timestep = timestep.to(hidden_states.dtype)

    if self.zero_cond_t:
        timestep = torch.cat([timestep, timestep * 0], dim=0)
        modulate_index = torch.tensor(
            [
                [0] * int(np.prod(sample[0])) + [1] * int(sum(np.prod(s) for s in sample[1:]))
                for sample in img_shapes
            ],
            device=timestep.device,
            dtype=torch.int,
        )
        if image_token_indices is not None:
            modulate_index = modulate_index.index_select(
                dim=1, index=image_token_indices.to(modulate_index.device)
            )
    else:
        modulate_index = None

    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
        encoder_hidden_states, encoder_hidden_states_mask
    )

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, hidden_states, additional_t_cond)
        if guidance is None
        else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
    )

    image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)
    if image_token_indices is not None and hasattr(self, "_slice_image_rotary_emb"):
        image_rotary_emb = self._slice_image_rotary_emb(image_rotary_emb, image_token_indices)

    block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
    if encoder_hidden_states_mask is not None:
        batch_size, image_seq_len = hidden_states.shape[:2]
        image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
        joint_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
        block_attention_kwargs["attention_mask"] = joint_attention_mask

    forward_type = "full"
    if enable:
        _ensure_taylorseer_runtime(self)
        cache_dic = self.taylorseer_cache_dic
        cur = self.taylorseer_current
        cur["step"] = step_index
        cur["num_steps"] = num_steps

        if branch_key == "branch_0":
            _cal_type(cache_dic, cur)
            forward_type = cur["type"]
            if step_index == num_steps - 1:
                forward_type = "full"
            self._firered_taylor_forward_type = forward_type
        else:
            forward_type = self._firered_taylor_forward_type or "full"

    used_taylor = False
    if enable and forward_type == "Taylor":
        cache_dic = self.taylorseer_cache_dic
        cur = self.taylorseer_current
        distance = int(step_index - cur["activated_steps"][-1])
        distance = max(distance, 1)
        layer_c = cache_dic["cache"][-1][stream][LAYER_IDX]
        enc_d = layer_c.get(MOD_ENC)
        hid_d = layer_c.get(MOD_HID)
        if enc_d is not None and hid_d is not None and len(enc_d) > 0 and len(hid_d) > 0:
            pred_e = _taylor_formula(enc_d, distance).to(encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
            pred_h = _taylor_formula(hid_d, distance).to(hidden_states.device, dtype=hidden_states.dtype)
            encoder_hidden_states = encoder_hidden_states + pred_e
            hidden_states = hidden_states + pred_h
            used_taylor = True
        else:
            forward_type = "full"

    if not used_taylor:
        ori_enc = encoder_hidden_states.detach().to(cache_device)
        ori_hid = hidden_states.detach().to(cache_device)

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    None,
                    temb,
                    image_rotary_emb,
                    block_attention_kwargs,
                    modulate_index,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=None,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=block_attention_kwargs,
                    modulate_index=modulate_index,
                )

            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        if enable and forward_type == "full":
            cur = self.taylorseer_current
            cur["layer"] = LAYER_IDX
            enc_res = (encoder_hidden_states - ori_enc.to(encoder_hidden_states.device)).detach().to(cache_device)
            hid_res = (hidden_states - ori_hid.to(hidden_states.device)).detach().to(cache_device)
            _derivative_approximation(self.taylorseer_cache_dic, cur, stream, MOD_ENC, enc_res)
            _derivative_approximation(self.taylorseer_cache_dic, cur, stream, MOD_HID, hid_res)
            acts = self.taylorseer_current["activated_steps"]
            if step_index == num_steps - 1 and (not acts or acts[-1] != step_index):
                acts.append(step_index)

    if self.zero_cond_t:
        temb = temb.chunk(2, dim=0)[0]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


def apply_taylorseer_monkey_patch() -> None:
    from firered.transformer import QwenImageTransformer2DModel as FireRedQwenImageTransformer2DModel

    FireRedQwenImageTransformer2DModel.forward = taylorseer_forward
    FireRedQwenImageTransformer2DModel.reset_taylorseer_state = reset_taylorseer_state

    try:
        from diffusers.models import QwenImageTransformer2DModel as DiffusersQwenImageTransformer2DModel

        if DiffusersQwenImageTransformer2DModel is not FireRedQwenImageTransformer2DModel:
            DiffusersQwenImageTransformer2DModel.forward = taylorseer_forward
            DiffusersQwenImageTransformer2DModel.reset_taylorseer_state = reset_taylorseer_state
    except Exception as exc:
        logger.warning("Skip diffusers QwenImageTransformer2DModel TaylorSeer patch: %s", exc)

    try:
        from firered_cubic.transformer import CubicQwenImageTransformer2DModel

        CubicQwenImageTransformer2DModel.forward = taylorseer_forward
        CubicQwenImageTransformer2DModel.reset_taylorseer_state = reset_taylorseer_state
    except Exception as exc:
        logger.warning("Skip firered_cubic TaylorSeer patch: %s", exc)


apply_taylorseer_monkey_patch()
