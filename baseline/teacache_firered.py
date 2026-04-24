"""
TeaCache for FireRed (`firered_exp/firered/transformer.py` 中的 `QwenImageTransformer2DModel`).

设计要点（对齐 `teacache_flux.py` 思路 + Qwen-Image-Edit 系常用实现）：
- 用首块 `img_mod` + `img_norm1` + `_modulate` 的特征作为步间相似度探针；
- 累积缩放后的相对 L1，超过阈值则完整重算主干，否则复用上一步的 encoder / image 残差；
- CFG 下同一步内 cond / uncond 两次调用用独立分支状态（branch_0 / branch_1）。

必须在加载的 **FireRed 本地 Transformer 类** 上打补丁；仅补丁 diffusers 内置类时，
`firered.transformer.QwenImageTransformer2DModel` 实例不会生效。
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import apply_lora_scale, deprecate, logging

from firered.transformer import compute_text_seq_len_from_mask

logger = logging.get_logger(__name__)

# Qwen-Image-Edit 系 TeaCache 常用多项式系数（与常见开源实现一致）
QWEN_IMAGE_EDIT_COEFFICIENTS = np.array(
    [-4.50000000e02, 2.80000000e02, -4.50000000e01, 3.20000000e00, -2.00000000e-02],
    dtype=np.float64,
)


def _poly1d(coefficients: np.ndarray, x: float) -> float:
    result = 0.0
    degree = len(coefficients) - 1
    for i, coeff in enumerate(coefficients):
        result += float(coeff) * (float(x) ** (degree - i))
    return result


def _make_branch_state() -> dict[str, Any]:
    return {
        "accumulated_rel_l1_distance": 0.0,
        "previous_modulated_input": None,
        "previous_hidden_residual": None,
        "previous_encoder_residual": None,
    }


def _reset_teacache_runtime(self) -> None:
    self.teacache_runtime = {
        "last_timestep": None,
        "step_index": 0,
        "call_index_for_step": -1,
    }
    self.teacache_states = {
        "branch_0": _make_branch_state(),
        "branch_1": _make_branch_state(),
    }


def _ensure_teacache_runtime(self) -> None:
    if not hasattr(self, "teacache_runtime") or not hasattr(self, "teacache_states"):
        _reset_teacache_runtime(self)


def reset_teacache_state(self) -> None:
    """在新一次生成前调用，清空 TeaCache 运行时状态。"""
    _reset_teacache_runtime(self)


def _update_teacache_runtime(self, timestep: torch.Tensor, num_steps: int) -> tuple[int, str]:
    """
    按「同一 timestep 下的调用次序」区分 CFG 分支（cond=branch_0, uncond=branch_1）。
    """
    _ensure_teacache_runtime(self)

    current_timestep = float(timestep.detach().flatten()[0].float().cpu().item())
    runtime = self.teacache_runtime
    last_timestep = runtime["last_timestep"]

    same_step = last_timestep is not None and abs(current_timestep - last_timestep) <= 1e-12

    if last_timestep is None:
        runtime["step_index"] = 0
        runtime["call_index_for_step"] = 0
    elif same_step:
        runtime["call_index_for_step"] += 1
    else:
        new_generation = current_timestep > last_timestep or runtime["step_index"] >= max(int(num_steps) - 1, 0)
        if new_generation:
            _reset_teacache_runtime(self)
            runtime = self.teacache_runtime
            runtime["step_index"] = 0
            runtime["call_index_for_step"] = 0
        else:
            runtime["step_index"] += 1
            runtime["call_index_for_step"] = 0

    runtime["last_timestep"] = current_timestep
    branch_key = f"branch_{min(runtime['call_index_for_step'], 1)}"
    return int(runtime["step_index"]), branch_key


def teacache_forward(
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
    """
    FireRed `QwenImageTransformer2DModel` / `CubicQwenImageTransformer2DModel` 的 TeaCache forward。

    `image_token_indices` 仅 Cubic 子模型使用；标准 FireRed 管线不传，保持默认 None 即可。
    """
    if txt_seq_lens is not None:
        deprecate(
            "txt_seq_lens",
            "0.39.0",
            "Passing `txt_seq_lens` is deprecated and will be removed in version 0.39.0. "
            "Please use `encoder_hidden_states_mask` instead.",
            standard_warn=False,
        )

    enable_teacache = bool(getattr(self, "enable_teacache", False))
    num_steps = int(getattr(self, "num_steps", 50))
    rel_l1_thresh = float(getattr(self, "rel_l1_thresh", 0.55))
    cache_device = getattr(self, "teacache_cache_device", hidden_states.device)
    coefficients = np.asarray(
        getattr(self, "teacache_coefficients", QWEN_IMAGE_EDIT_COEFFICIENTS),
        dtype=np.float64,
    )

    step_index, branch_key = _update_teacache_runtime(self, timestep, num_steps)
    branch_state = self.teacache_states[branch_key]

    hidden_states = self.img_in(hidden_states)

    # Cubic：与 `firered_cubic/transformer.py` 一致，在 img_in 之后规范化索引
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

    should_calc = True
    if enable_teacache:
        inp = hidden_states.clone()
        temb_ = temb.clone()
        block0 = self.transformer_blocks[0]
        img_mod_params = block0.img_mod(temb_)
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        img_normed = block0.img_norm1(inp)
        modulated_inp, _gate_msa = block0._modulate(img_normed, img_mod1, modulate_index)

        if step_index == 0 or step_index == num_steps - 1:
            should_calc = True
            branch_state["accumulated_rel_l1_distance"] = 0.0
        elif branch_state["previous_modulated_input"] is None:
            should_calc = True
        else:
            prev_modulated_inp = branch_state["previous_modulated_input"]
            try:
                rel_l1 = (
                    (modulated_inp - prev_modulated_inp).abs().mean()
                    / (prev_modulated_inp.abs().mean() + 1e-8)
                ).detach().float().cpu().item()
                scaled_rel_l1 = abs(_poly1d(coefficients, rel_l1))
                branch_state["accumulated_rel_l1_distance"] += scaled_rel_l1
                should_calc = branch_state["accumulated_rel_l1_distance"] >= rel_l1_thresh
                if should_calc:
                    branch_state["accumulated_rel_l1_distance"] = 0.0
            except Exception:
                should_calc = True
                branch_state["accumulated_rel_l1_distance"] = 0.0

        branch_state["previous_modulated_input"] = modulated_inp.detach().clone()

    block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
    if encoder_hidden_states_mask is not None:
        batch_size, image_seq_len = hidden_states.shape[:2]
        image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
        joint_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
        block_attention_kwargs["attention_mask"] = joint_attention_mask

    used_cache = False
    if enable_teacache and not should_calc:
        prev_enc_res = branch_state["previous_encoder_residual"]
        prev_hid_res = branch_state["previous_hidden_residual"]
        if (
            prev_enc_res is not None
            and prev_hid_res is not None
            and tuple(prev_enc_res.shape) == tuple(encoder_hidden_states.shape)
            and tuple(prev_hid_res.shape) == tuple(hidden_states.shape)
        ):
            encoder_hidden_states = encoder_hidden_states + prev_enc_res.to(encoder_hidden_states.device)
            hidden_states = hidden_states + prev_hid_res.to(hidden_states.device)
            used_cache = True
        else:
            should_calc = True

    if not used_cache:
        ori_encoder_hidden_states = encoder_hidden_states.detach().to(cache_device)
        ori_hidden_states = hidden_states.detach().to(cache_device)

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

        if enable_teacache:
            branch_state["previous_encoder_residual"] = (
                encoder_hidden_states.detach().to(cache_device) - ori_encoder_hidden_states
            )
            branch_state["previous_hidden_residual"] = hidden_states.detach().to(cache_device) - ori_hidden_states

    if self.zero_cond_t:
        temb = temb.chunk(2, dim=0)[0]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


def apply_teacache_monkey_patch() -> None:
    """将 TeaCache forward 绑定到 FireRed / diffusers / Cubic 的 Transformer 类（仅绑定不同类对象）。"""
    from firered.transformer import QwenImageTransformer2DModel as FireRedQwenImageTransformer2DModel

    FireRedQwenImageTransformer2DModel.forward = teacache_forward
    FireRedQwenImageTransformer2DModel.reset_teacache_state = reset_teacache_state

    try:
        from diffusers.models import QwenImageTransformer2DModel as DiffusersQwenImageTransformer2DModel

        if DiffusersQwenImageTransformer2DModel is not FireRedQwenImageTransformer2DModel:
            DiffusersQwenImageTransformer2DModel.forward = teacache_forward
            DiffusersQwenImageTransformer2DModel.reset_teacache_state = reset_teacache_state
    except Exception as exc:
        logger.warning("Skip diffusers QwenImageTransformer2DModel patch: %s", exc)

    try:
        from firered_cubic.transformer import CubicQwenImageTransformer2DModel

        CubicQwenImageTransformer2DModel.forward = teacache_forward
        CubicQwenImageTransformer2DModel.reset_teacache_state = reset_teacache_state
    except Exception as exc:
        logger.warning("Skip firered_cubic patch: %s", exc)


apply_teacache_monkey_patch()


# ---------------------------------------------------------------------------
# 使用说明（与 `teacache_flux.py` 末尾类似，按需复制到推理脚本）
# ---------------------------------------------------------------------------
"""
from baseline.teacache_firered import apply_teacache_monkey_patch  # 导入即已 patch，可省略重复调用
# apply_teacache_monkey_patch()

transformer = pipeline.transformer
transformer.enable_teacache = True
transformer.num_steps = num_inference_steps
transformer.rel_l1_thresh = 0.55
transformer.teacache_cache_device = transformer.device if hasattr(transformer, "device") else "cuda"
transformer.teacache_coefficients = [-4.5e2, 2.8e2, -4.5e1, 3.2, -2.0e-2]

if hasattr(transformer, "reset_teacache_state"):
    transformer.reset_teacache_state()
"""
