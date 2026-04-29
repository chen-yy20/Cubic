"""
PAB (Pyramid Attention Broadcast) for FireRed (`firered/transformer.py`).

设计要点（对齐 teacache_firered.py 的风格）：
- 对每个 QwenImageTransformerBlock 的 joint attention (block.attn) 注入 PAB 缓存；
- 在 warmup_steps ~ cooldown_steps 范围内，按 pab_skip_range 间隔复用 attention 输出；
- CFG 下同一步内 cond/uncond 两次调用用独立分支缓存（branch_0 / branch_1）；
- 第一次 forward 时动态 patch block.attn（实例级别，不污染类定义）。

使用方式：
    import baseline.PAB_firered  # 导入即自动 monkey patch FireRed Transformer 类

    transformer = pipeline.transformer
    transformer.enable_pab = True
    transformer.num_steps = num_inference_steps
    transformer.pab_warmup_steps = 5              # 从第几步开始启用 PAB（含）
    transformer.pab_cooldown_steps = steps - 5    # 从第几步停止 PAB（不含，最后几步全量计算）
    transformer.pab_skip_range = 2                # attention 复用间隔（每隔几步重算一次）
    if hasattr(transformer, 'reset_pab_state'):
        transformer.reset_pab_state()
"""

from __future__ import annotations

import functools
import weakref
from typing import Any, Union

import numpy as np
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import apply_lora_scale, deprecate, logging

from firered.transformer import compute_text_seq_len_from_mask

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# 运行时状态管理
# ---------------------------------------------------------------------------

def _make_pab_runtime() -> dict[str, Any]:
    return {
        "last_timestep": None,
        "step_index": 0,
        "call_index_for_step": -1,
    }


def _reset_pab_runtime(self) -> None:
    self._pab_runtime = _make_pab_runtime()
    self._pab_cache: dict[str, Any] = {}


def _ensure_pab_runtime(self) -> None:
    if not hasattr(self, "_pab_runtime"):
        _reset_pab_runtime(self)


def reset_pab_state(self) -> None:
    """在新一次生成前调用，清空 PAB 运行时状态与 attention 缓存。"""
    _reset_pab_runtime(self)


def _update_pab_runtime(self, timestep: torch.Tensor, num_steps: int) -> tuple[int, str]:
    """
    按「同一 timestep 下的调用次序」区分 CFG 分支（cond=branch_0, uncond=branch_1）。
    逻辑与 teacache_firered._update_teacache_runtime 完全一致。
    """
    _ensure_pab_runtime(self)

    current_timestep = float(timestep.detach().flatten()[0].float().cpu().item())
    runtime = self._pab_runtime
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
            _reset_pab_runtime(self)
            runtime = self._pab_runtime
            runtime["step_index"] = 0
            runtime["call_index_for_step"] = 0
        else:
            runtime["step_index"] += 1
            runtime["call_index_for_step"] = 0

    runtime["last_timestep"] = current_timestep
    branch_key = f"branch_{min(runtime['call_index_for_step'], 1)}"
    return int(runtime["step_index"]), branch_key


# ---------------------------------------------------------------------------
# Block 级别的 PAB attention patch（实例级别，懒初始化）
# ---------------------------------------------------------------------------

def _patch_pab_blocks(transformer) -> None:
    """
    在 transformer 实例上为每个 block.attn 注入 PAB 缓存逻辑。
    使用弱引用避免循环引用，并通过闭包传递 block_idx。
    """
    transformer_ref = weakref.ref(transformer)

    for block_idx, block in enumerate(transformer.transformer_blocks):
        if hasattr(block.attn, "_pab_original_forward"):
            continue  # 已 patch，跳过

        block.attn._pab_original_forward = block.attn.forward
        original_fn = block.attn._pab_original_forward

        def _make_pab_attn_forward(orig_fn, blk_idx, tr_ref):
            @functools.wraps(orig_fn)
            def pab_attn_forward(*args, **kwargs):
                tr = tr_ref()
                if tr is None or not getattr(tr, "enable_pab", False):
                    return orig_fn(*args, **kwargs)

                num_steps = int(getattr(tr, "num_steps", 50))
                warmup_steps = int(getattr(tr, "pab_warmup_steps", 5))
                cooldown_steps = int(getattr(tr, "pab_cooldown_steps", num_steps - 5))
                skip_range = int(getattr(tr, "pab_skip_range", 2))

                step = int(getattr(tr, "_pab_current_step", 0))
                branch_key = str(getattr(tr, "_pab_current_branch", "branch_0"))

                if not hasattr(tr, "_pab_cache"):
                    tr._pab_cache = {}

                cache_key = f"{branch_key}_block{blk_idx}"
                in_active_range = warmup_steps <= step < cooldown_steps

                # 判断本步是否跳过 attention 计算（复用缓存）
                should_reuse = (
                    in_active_range
                    and (step - warmup_steps) % skip_range != 0
                    and cache_key in tr._pab_cache
                )

                if should_reuse:
                    return tr._pab_cache[cache_key]

                output = orig_fn(*args, **kwargs)

                # 只在有效范围内存缓存
                if in_active_range:
                    tr._pab_cache[cache_key] = output

                return output

            return pab_attn_forward

        block.attn.forward = _make_pab_attn_forward(original_fn, block_idx, transformer_ref)

    logger.info(
        "[PAB] Patched %d transformer_blocks on instance %s",
        len(transformer.transformer_blocks),
        type(transformer).__name__,
    )


def _unpatch_pab_blocks(transformer) -> None:
    """恢复每个 block.attn 的原始 forward（用于 unpatch / 调试）。"""
    for block in transformer.transformer_blocks:
        if hasattr(block.attn, "_pab_original_forward"):
            block.attn.forward = block.attn._pab_original_forward
            del block.attn._pab_original_forward
    transformer._pab_blocks_patched = False
    logger.info("[PAB] Unpatched transformer_blocks on instance %s", type(transformer).__name__)


# ---------------------------------------------------------------------------
# PAB-aware transformer forward
# ---------------------------------------------------------------------------

@apply_lora_scale("attention_kwargs")
def pab_forward(
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
    FireRed `QwenImageTransformer2DModel` / `CubicQwenImageTransformer2DModel` 的 PAB forward。

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

    enable_pab = bool(getattr(self, "enable_pab", False))
    num_steps = int(getattr(self, "num_steps", 50))

    # ------------------------------------------------------------------
    # 1. 更新 PAB 运行时状态（step 索引 & CFG 分支）
    # ------------------------------------------------------------------
    if enable_pab:
        step_index, branch_key = _update_pab_runtime(self, timestep, num_steps)
        self._pab_current_step = step_index
        self._pab_current_branch = branch_key

        # 懒初始化：在第一次 forward 时 patch block.attn（实例级别）
        if not getattr(self, "_pab_blocks_patched", False):
            _patch_pab_blocks(self)
            self._pab_blocks_patched = True

    # ------------------------------------------------------------------
    # 2. 标准前向处理（与 firered/transformer.py 完全一致）
    # ------------------------------------------------------------------
    hidden_states = self.img_in(hidden_states)

    # Cubic 子模型支持（与 firered_cubic/transformer.py 一致）
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

    # ------------------------------------------------------------------
    # 3. 遍历 transformer blocks
    #    block.attn 已被 PAB patch 拦截，自动处理缓存复用逻辑
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 4. 输出层
    # ------------------------------------------------------------------
    if self.zero_cond_t:
        temb = temb.chunk(2, dim=0)[0]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


# ---------------------------------------------------------------------------
# Monkey patch 入口
# ---------------------------------------------------------------------------

def apply_pab_monkey_patch() -> None:
    """将 PAB forward 绑定到 FireRed / diffusers / Cubic 的 Transformer 类。"""
    from firered.transformer import QwenImageTransformer2DModel as FireRedQwenImageTransformer2DModel

    FireRedQwenImageTransformer2DModel.forward = pab_forward
    FireRedQwenImageTransformer2DModel.reset_pab_state = reset_pab_state

    try:
        from diffusers.models import QwenImageTransformer2DModel as DiffusersQwenImageTransformer2DModel

        if DiffusersQwenImageTransformer2DModel is not FireRedQwenImageTransformer2DModel:
            DiffusersQwenImageTransformer2DModel.forward = pab_forward
            DiffusersQwenImageTransformer2DModel.reset_pab_state = reset_pab_state
    except Exception as exc:
        logger.warning("Skip diffusers QwenImageTransformer2DModel patch: %s", exc)

    try:
        from firered_cubic.transformer import CubicQwenImageTransformer2DModel

        CubicQwenImageTransformer2DModel.forward = pab_forward
        CubicQwenImageTransformer2DModel.reset_pab_state = reset_pab_state
    except Exception as exc:
        logger.warning("Skip firered_cubic patch: %s", exc)


apply_pab_monkey_patch()
