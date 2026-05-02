"""
Jano (Joint Adaptive Network Optimization) for FireRed QwenImageTransformer2DModel.

正确的 KV Cache 实现：
  - 在 Attention Processor 层面实现 KV Cache
  - 静态 token 的 K/V 从缓存读取，只重新计算动态 token 的 K/V
  - 选中 token 的 Q 与全量 K/V（缓存的+新计算的）进行 attention
  - 所有 token 都参与计算，但通过 KV Cache 减少重复计算

FireRed 序列设计：
  latent_model_input = cat([noise_latent, ref_latent], dim=1)
  img_shapes[b] = [(1, noise_h, noise_w), (1, ref1_h, ref1_w), ...]

使用方式（从 Cubic 根目录运行，PYTHONPATH=.）：
    import baseline.jano_firered            # 导入即触发 monkey patch

    transformer = pipeline.transformer
    transformer.enable_jano = True
    transformer.num_steps   = num_inference_steps
    transformer.jano_warmup_steps    = 5
    transformer.jano_cooldown_steps  = 5
    transformer.jano_static_interval = 6
    transformer.jano_medium_interval = 3
    transformer.jano_static_thresh   = 0.2
    transformer.jano_medium_thresh   = 0.6
    transformer.jano_ref_thresh      = 0.5
    transformer.jano_analyze_block_size = (1, 4, 4)
    if hasattr(transformer, "reset_jano_state"):
        transformer.reset_jano_state()
"""

from __future__ import annotations

import functools
import os
import weakref
from math import prod
from typing import Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import apply_lora_scale, deprecate, logging

from firered.transformer import (
    compute_text_seq_len_from_mask,
    QwenDoubleStreamAttnProcessor2_0,
    apply_rotary_emb_qwen,
    dispatch_attention_fn,
)
from firered_exp.jano.dynamic_analyzer import DynamicAnalyzer
from firered_exp.jano.firered_mask_manager import (
    FireRedMaskManager,
    generate_ref_sequence_mask,
)
from utils.envs import GlobalEnv

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# 运行时状态管理
# ---------------------------------------------------------------------------

def _make_jano_runtime() -> dict[str, Any]:
    return {
        "last_timestep": None,
        "step_index": 0,
        "call_index_for_step": -1,
    }


def _reset_jano_runtime(self) -> None:
    self._jano_runtime = _make_jano_runtime()


def _ensure_jano_runtime(self) -> None:
    if not hasattr(self, "_jano_runtime"):
        _reset_jano_runtime(self)


def _update_jano_runtime(self, timestep: torch.Tensor, num_steps: int) -> tuple[int, str]:
    """推断当前 denoising step 索引与 CFG 分支。"""
    _ensure_jano_runtime(self)

    cur_ts = float(timestep.detach().flatten()[0].float().cpu().item())
    rt = self._jano_runtime
    last_ts = rt["last_timestep"]
    same = last_ts is not None and abs(cur_ts - last_ts) <= 1e-12

    if last_ts is None:
        rt["step_index"] = 0
        rt["call_index_for_step"] = 0
    elif same:
        rt["call_index_for_step"] += 1
    else:
        new_gen = cur_ts > last_ts or rt["step_index"] >= max(int(num_steps) - 1, 0)
        if new_gen:
            _reset_jano_runtime(self)
            rt = self._jano_runtime
            rt["step_index"] = 0
            rt["call_index_for_step"] = 0
        else:
            rt["step_index"] += 1
            rt["call_index_for_step"] = 0

    rt["last_timestep"] = cur_ts
    branch_key = f"branch_{min(rt['call_index_for_step'], 1)}"
    return int(rt["step_index"]), branch_key


# ---------------------------------------------------------------------------
# Jano GlobalEnv 初始化
# ---------------------------------------------------------------------------

def _setup_jano_global_env(self) -> None:
    """在第一次 forward 时将 Jano 所需的 GlobalEnv 参数写入。"""
    block_size = tuple(getattr(self, "jano_analyze_block_size", (1, 4, 4)))
    warmup_steps = int(getattr(self, "jano_warmup_steps", 5))
    t_weight = float(getattr(self, "jano_t_weight", 0.7))
    d_strength = float(getattr(self, "jano_d_strength", 0.8))
    d_distance = int(getattr(self, "jano_d_distance", 1))
    tag = str(getattr(self, "jano_tag", "jano_firered"))
    save_dir = str(getattr(self, "jano_save_dir", "./results/jano"))

    GlobalEnv.set_envs("analyze_block_size", block_size)
    GlobalEnv.set_envs("warmup_steps", warmup_steps)
    GlobalEnv.set_envs("t_weight", t_weight)
    GlobalEnv.set_envs("d_strength", d_strength)
    GlobalEnv.set_envs("d_distance", d_distance)
    GlobalEnv.set_envs("tag", tag)
    GlobalEnv.set_envs("save_dir", save_dir)
    GlobalEnv.set_envs("enable_jano", True)

    os.makedirs(save_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# 噪声 latent 空间形状解析
# ---------------------------------------------------------------------------

def _parse_img_shapes(img_shapes: list) -> tuple[int, int, int, list[tuple[int, int, int]]]:
    """从 img_shapes 解析噪声与参考图的空间尺寸。"""
    sample = img_shapes[0]
    noise_shape = sample[0]
    ref_shapes = list(sample[1:])
    noise_h, noise_w = int(noise_shape[1]), int(noise_shape[2])
    noise_seq_len = int(prod(noise_shape))
    return noise_h, noise_w, noise_seq_len, ref_shapes


def _ref_info(ref_shapes: list) -> tuple[int, int, int]:
    """从 ref_shapes 计算第一张参考图的空间尺寸及总 ref_seq_len。"""
    if not ref_shapes:
        return 0, 0, 0
    ref1_h = int(ref_shapes[0][1])
    ref1_w = int(ref_shapes[0][2])
    ref_seq_len = sum(int(prod(s)) for s in ref_shapes)
    return ref1_h, ref1_w, ref_seq_len


# ---------------------------------------------------------------------------
# Jano 状态初始化
# ---------------------------------------------------------------------------

def _ensure_jano_state(self, noise_h: int, noise_w: int,
                        noise_seq_len: int, ref_seq_len: int,
                        num_steps: int, in_channels: int) -> FireRedMaskManager:
    """懒初始化：若形状不变则复用已有状态。"""
    warmup_steps = int(getattr(self, "jano_warmup_steps", 5))
    cooldown_steps = int(getattr(self, "jano_cooldown_steps", 5))
    static_interval = int(getattr(self, "jano_static_interval", 6))
    medium_interval = int(getattr(self, "jano_medium_interval", 3))

    shape_key = (noise_h, noise_w, noise_seq_len, ref_seq_len)

    if not hasattr(self, "_jano_mm") or self._jano_shape_key != shape_key:
        _setup_jano_global_env(self)

        mm = FireRedMaskManager(
            noise_seq_len=noise_seq_len,
            ref_seq_len=ref_seq_len,
            num_inference_steps=num_steps,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            static_interval=static_interval,
            medium_interval=medium_interval,
        )
        self._jano_mm = mm
        self._jano_shape_key = shape_key
        self._jano_in_channels = in_channels

        da = DynamicAnalyzer(T=1, H=noise_h, W=noise_w, C=in_channels)
        self._jano_da = da

    return self._jano_mm


# ---------------------------------------------------------------------------
# warmup：收集噪声 latent
# ---------------------------------------------------------------------------

def _collect_noise_latent(
    self,
    raw_hidden_states: torch.Tensor,
    noise_seq_len: int,
    noise_h: int,
    noise_w: int,
    in_channels: int,
    branch_key: str,
) -> None:
    """仅对 branch_0（cond 路）收集噪声 latent。"""
    if branch_key != "branch_0":
        return

    da: DynamicAnalyzer = self._jano_da
    if not isinstance(da.stored_features, list):
        return

    noise_packed = raw_hidden_states[:, :noise_seq_len, :]
    noise_spatial = (
        noise_packed[0]
        .reshape(noise_h, noise_w, in_channels)
        .permute(2, 0, 1)
        .unsqueeze(1)
    )
    da.stored_features.append(noise_spatial)


# ---------------------------------------------------------------------------
# 分析阶段：生成 noise mask + ref mask
# ---------------------------------------------------------------------------

def _finalize_analysis(
    self,
    raw_hidden_states: torch.Tensor,
    noise_seq_len: int,
    noise_h: int,
    noise_w: int,
    ref_seq_len: int,
    ref_shapes: list,
    in_channels: int,
    enhanced_combined: torch.Tensor,
) -> None:
    """在 warmup 结束后，生成噪声掩码和参考图掩码。"""
    mm: FireRedMaskManager = self._jano_mm
    da: DynamicAnalyzer = self._jano_da

    static_thresh = float(getattr(self, "jano_static_thresh", 0.2))
    medium_thresh = float(getattr(self, "jano_medium_thresh", 0.6))
    ref_thresh = float(getattr(self, "jano_ref_thresh", 0.5))
    block_size = tuple(getattr(self, "jano_analyze_block_size", (1, 4, 4)))
    block_h, block_w = int(block_size[1]), int(block_size[2])

    da.stored_features = None

    # 噪声 mask
    bm = da.block_manager
    bt, bh, bw = bm.block_size
    nt, nh, nw = bm.nt, bm.nh, bm.nw

    ec_cpu = enhanced_combined.cpu()
    block_mask = torch.ones(bm.total_blocks, dtype=torch.int8)
    medium_cond = (ec_cpu > static_thresh) & (ec_cpu <= medium_thresh)
    high_cond = ec_cpu > medium_thresh
    block_mask = torch.where(medium_cond, torch.tensor(2, dtype=torch.int8), block_mask)
    block_mask = torch.where(high_cond, torch.tensor(3, dtype=torch.int8), block_mask)

    block_mask_3d = block_mask.reshape(nt, nh, nw)
    latent_mask = torch.zeros((noise_h, noise_w), dtype=torch.int8)
    for hi in range(nh):
        for wi in range(nw):
            val = block_mask_3d[0, hi, wi].item()
            r0 = hi * bh
            r1 = min(r0 + bh, noise_h)
            c0 = wi * bw
            c1 = min(c0 + bw, noise_w)
            if r0 < noise_h and c0 < noise_w:
                latent_mask[r0:r1, c0:c1] = val
    noise_seq_mask = latent_mask.flatten()
    mm.set_noise_sequence_mask(noise_seq_mask)

    # 参考图 mask
    if ref_seq_len > 0:
        ref1_h = int(ref_shapes[0][1])
        ref1_w = int(ref_shapes[0][2])
        ref_packed = raw_hidden_states[:, noise_seq_len:noise_seq_len + ref1_h * ref1_w, :]
        ref_seq_mask = generate_ref_sequence_mask(
            ref_packed=ref_packed,
            ref_h=ref1_h,
            ref_w=ref1_w,
            block_h=block_h,
            block_w=block_w,
            ref_thresh=ref_thresh,
        ).cpu()

        total_ref_seq_mask = ref_seq_mask
        for extra_shape in ref_shapes[1:]:
            extra_len = int(prod(extra_shape))
            extra_mask = torch.full((extra_len,), 3, dtype=torch.int8)
            total_ref_seq_mask = torch.cat([total_ref_seq_mask, extra_mask])
        mm.set_ref_sequence_mask(total_ref_seq_mask)
    else:
        mm.masks_ready = True


# ---------------------------------------------------------------------------
# Jano KV Cache Attention Processor
# ---------------------------------------------------------------------------

class JanoKVCacheAttnProcessor:
    """
    Jano KV Cache Attention Processor。
    在标准 QwenDoubleStreamAttnProcessor2_0 基础上增加 KV Cache 逻辑：
    - 静态 token 的 K/V 从缓存读取
    - 动态 token 重新计算 K/V
    - 所有 token 的 Q 都重新计算
    """

    def __init__(self, block_idx: int, transformer_ref: weakref.ref):
        self.block_idx = block_idx
        self.transformer_ref = transformer_ref
        self._base_processor = QwenDoubleStreamAttnProcessor2_0()

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ):
        tr = self.transformer_ref()
        if tr is None or not getattr(tr, "enable_jano", False):
            # 不使用 Jano，直接调用基础 processor
            return self._base_processor(
                attn, hidden_states, encoder_hidden_states,
                encoder_hidden_states_mask, attention_mask, image_rotary_emb
            )

        mm: FireRedMaskManager = getattr(tr, "_jano_mm", None)
        if mm is None or not mm.masks_ready:
            return self._base_processor(
                attn, hidden_states, encoder_hidden_states,
                encoder_hidden_states_mask, attention_mask, image_rotary_emb
            )

        branch_key = str(getattr(tr, "_jano_current_branch", "branch_0"))

        # step_level 与 compute_mask 在 jano_forward 里每 denoise 步只更新一次，避免
        # 60 层 × 每步重复 update_step_level（内含 cudaMem 查询与 tqdm 写）导致比 ori 更慢。
        step_level = int(getattr(tr, "_jano_step_level", 0))

        # L0/L3：全量计算并刷新 K/V 缓存
        if step_level in (0, 3):
            out = self._base_processor(
                attn, hidden_states, encoder_hidden_states,
                encoder_hidden_states_mask, attention_mask, image_rotary_emb
            )
            self._cache_kv(tr, attn, hidden_states, encoder_hidden_states, branch_key)
            return out

        compute_mask = getattr(tr, "_jano_compute_mask_dev", None)
        dyn_idx = getattr(tr, "_jano_dynamic_idx", None)
        if compute_mask is None or dyn_idx is None:
            return self._base_processor(
                attn, hidden_states, encoder_hidden_states,
                encoder_hidden_states_mask, attention_mask, image_rotary_emb
            )

        return self._forward_with_kv_cache(
            attn, hidden_states, encoder_hidden_states,
            encoder_hidden_states_mask, attention_mask, image_rotary_emb,
            compute_mask, dyn_idx, branch_key
        )

    def _cache_kv(self, tr, attn, hidden_states, encoder_hidden_states, branch_key):
        """缓存当前 step 的 K/V。"""
        if not hasattr(tr, "_jano_kv_cache"):
            tr._jano_kv_cache = {}

        with torch.no_grad():
            img_key = attn.to_k(hidden_states)
            img_value = attn.to_v(hidden_states)
            txt_key = attn.add_k_proj(encoder_hidden_states)
            txt_value = attn.add_v_proj(encoder_hidden_states)

        cache_key = f"{branch_key}_block{self.block_idx}"
        tr._jano_kv_cache[cache_key] = {
            "img_key": img_key.detach(),
            "img_value": img_value.detach(),
            "txt_key": txt_key.detach(),
            "txt_value": txt_value.detach(),
        }

    def _forward_with_kv_cache(
        self, attn, hidden_states, encoder_hidden_states,
        encoder_hidden_states_mask, attention_mask, image_rotary_emb,
        compute_mask, dyn_idx: torch.Tensor, branch_key
    ):
        """使用 KV Cache 执行 attention。"""
        tr = self.transformer_ref()
        cache_key = f"{branch_key}_block{self.block_idx}"
        cached = tr._jano_kv_cache.get(cache_key) if hasattr(tr, "_jano_kv_cache") else None

        seq_txt = encoder_hidden_states.shape[1]
        dev = hidden_states.device

        # 1. 计算 Q（所有 token）
        img_query = attn.to_q(hidden_states)
        txt_query = attn.add_q_proj(encoder_hidden_states)

        # 2. 获取 K/V（静态的从缓存，动态的重新计算）
        if cached is not None:
            n_dyn = int(dyn_idx.numel())
            total_img = int(compute_mask.numel())
            # 动态占比极高时，布尔切片 + clone 合并往往比单次整段 Linear 更慢，直接全算
            if n_dyn == 0 or n_dyn * 10 > total_img * 9:
                img_key = attn.to_k(hidden_states)
                img_value = attn.to_v(hidden_states)
            else:
                img_key_cached = cached["img_key"]
                img_val_cached = cached["img_value"]
                if img_key_cached.device != dev:
                    img_key_cached = img_key_cached.to(dev, non_blocking=True)
                    img_val_cached = img_val_cached.to(dev, non_blocking=True)

                # index_select 比 hidden_states[:, bool_mask] 更易形成连续 GEMM
                h_sel = hidden_states.index_select(1, dyn_idx)
                img_key_dynamic = attn.to_k(h_sel)
                img_value_dynamic = attn.to_v(h_sel)

                img_key = torch.empty_like(img_key_cached)
                img_key.copy_(img_key_cached)
                img_key.index_copy_(1, dyn_idx, img_key_dynamic)

                img_value = torch.empty_like(img_val_cached)
                img_value.copy_(img_val_cached)
                img_value.index_copy_(1, dyn_idx, img_value_dynamic)

            txt_key = attn.add_k_proj(encoder_hidden_states)
            txt_value = attn.add_v_proj(encoder_hidden_states)
        else:
            # 无缓存，全量计算
            img_key = attn.to_k(hidden_states)
            img_value = attn.to_v(hidden_states)
            txt_key = attn.add_k_proj(encoder_hidden_states)
            txt_value = attn.add_v_proj(encoder_hidden_states)

        # 3. 更新缓存（为下一步准备）
        if hasattr(tr, "_jano_kv_cache"):
            tr._jano_kv_cache[cache_key] = {
                "img_key": img_key.detach(),
                "img_value": img_value.detach(),
                "txt_key": txt_key.detach(),
                "txt_value": txt_value.detach(),
            }

        # 4. 执行标准 attention 流程
        # Reshape for multi-head
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Joint attention
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        joint_hidden_states = dispatch_attention_fn(
            joint_query, joint_key, joint_value,
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            backend=QwenDoubleStreamAttnProcessor2_0._attention_backend,
            parallel_config=QwenDoubleStreamAttnProcessor2_0._parallel_config,
        )

        joint_hidden_states = joint_hidden_states.flatten(2, 3).to(joint_query.dtype)

        # Split outputs
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        # Output projections
        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())

        return img_attn_output, txt_attn_output


# ---------------------------------------------------------------------------
# Patch attention processors
# ---------------------------------------------------------------------------

def _patch_jano_attn_processors(transformer) -> None:
    """为每个 block 的 attn 设置 Jano KV Cache processor。"""
    transformer_ref = weakref.ref(transformer)

    for block_idx, block in enumerate(transformer.transformer_blocks):
        # 保存原始 processor
        if not hasattr(block.attn, "_jano_original_processor"):
            block.attn._jano_original_processor = block.attn.processor

        # 设置 Jano processor
        janoprocessor = JanoKVCacheAttnProcessor(block_idx, transformer_ref)
        block.attn.processor = janoprocessor

    logger.info(
        "[Jano] Patched %d attention processors with KV Cache",
        len(transformer.transformer_blocks),
    )


def _unpatch_jano_attn_processors(transformer) -> None:
    """恢复原始 attention processors。"""
    for block in transformer.transformer_blocks:
        if hasattr(block.attn, "_jano_original_processor"):
            block.attn.processor = block.attn._jano_original_processor
            del block.attn._jano_original_processor
    transformer._jano_attn_patched = False
    logger.info("[Jano] Unpatched attention processors")


# ---------------------------------------------------------------------------
# Jano-aware transformer forward
# ---------------------------------------------------------------------------

@apply_lora_scale("attention_kwargs")
def jano_forward(
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
    Jano-enhanced forward for FireRed QwenImageTransformer2DModel。
    使用 Attention Processor 层面的 KV Cache。
    """
    if txt_seq_lens is not None:
        deprecate("txt_seq_lens", "0.39.0", "deprecated", standard_warn=False)

    enable_jano = bool(getattr(self, "enable_jano", False))
    num_steps = int(getattr(self, "num_steps", 50))

    # 解析形状
    in_channels = hidden_states.shape[-1]

    if enable_jano and img_shapes is not None:
        noise_h, noise_w, noise_seq_len, ref_shapes = _parse_img_shapes(img_shapes)
        ref1_h, ref1_w, ref_seq_len = _ref_info(ref_shapes)
    else:
        noise_seq_len = hidden_states.shape[1]
        ref_seq_len = 0
        noise_h = noise_w = 0
        ref_shapes = []

    # Jano 运行时更新
    raw_hidden_states = hidden_states

    if enable_jano:
        step_index, branch_key = _update_jano_runtime(self, timestep, num_steps)
        self._jano_current_step = step_index
        self._jano_current_branch = branch_key

        # 初始化 KV Cache
        if not hasattr(self, "_jano_kv_cache"):
            self._jano_kv_cache = {}

        # 懒初始化 attention processors
        if not getattr(self, "_jano_attn_patched", False):
            _patch_jano_attn_processors(self)
            self._jano_attn_patched = True

        warmup_steps = int(getattr(self, "jano_warmup_steps", 5))
        mm = _ensure_jano_state(
            self, noise_h, noise_w, noise_seq_len, ref_seq_len,
            num_steps, in_channels,
        )

        # warmup：收集噪声 latent
        if step_index < warmup_steps:
            _collect_noise_latent(
                self, raw_hidden_states,
                noise_seq_len, noise_h, noise_w, in_channels, branch_key,
            )
        elif step_index == warmup_steps and branch_key == "branch_0" and not mm.masks_ready:
            da: DynamicAnalyzer = self._jano_da
            enhanced_combined = da.analyze()
            if enhanced_combined is not None:
                _finalize_analysis(
                    self, raw_hidden_states,
                    noise_seq_len, noise_h, noise_w,
                    ref_seq_len, ref_shapes, in_channels,
                    enhanced_combined,
                )

    # 标准 FireRed 前处理（所有 token 参与）
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
            device=timestep.device, dtype=torch.int,
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

    image_rotary_emb = self.pos_embed(
        img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device
    )
    if image_token_indices is not None and hasattr(self, "_slice_image_rotary_emb"):
        image_rotary_emb = self._slice_image_rotary_emb(image_rotary_emb, image_token_indices)

    block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
    if encoder_hidden_states_mask is not None:
        batch_size, image_seq_len = hidden_states.shape[:2]
        image_mask = torch.ones(
            (batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device
        )
        joint_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
        block_attention_kwargs["attention_mask"] = joint_attention_mask

    # 每 denoise 步只更新一次 step_level / 掩码 GPU 张量，避免 60×attention 重复
    # update_step_level（含 cuda 显存查询与 tqdm）与 get_compute_mask().to(device)。
    if enable_jano:
        mm = getattr(self, "_jano_mm", None)
        if mm is not None:
            mm.update_step_level(int(self._jano_current_step))
            self._jano_step_level = int(mm.step_level)
            if mm.masks_ready and mm.step_level in (1, 2):
                cm = mm.get_compute_mask()
                if cm is not None:
                    cm_dev = cm.to(hidden_states.device, non_blocking=True)
                    self._jano_compute_mask_dev = cm_dev
                    if cm_dev.dim() == 1:
                        self._jano_dynamic_idx = torch.nonzero(cm_dev, as_tuple=False).view(-1).long()
                    else:
                        self._jano_dynamic_idx = (
                            torch.nonzero(cm_dev[0], as_tuple=False).squeeze(-1).long()
                        )
                else:
                    self._jano_compute_mask_dev = None
                    self._jano_dynamic_idx = None
            else:
                self._jano_compute_mask_dev = None
                self._jano_dynamic_idx = None
        else:
            self._jano_step_level = 0
            self._jano_compute_mask_dev = None
            self._jano_dynamic_idx = None

    # Transformer blocks（KV Cache 在 attention processor 内部处理）
    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, encoder_hidden_states, None,
                temb, image_rotary_emb, block_attention_kwargs, modulate_index,
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

    # 输出层
    if self.zero_cond_t:
        temb = temb.chunk(2, dim=0)[0]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


# ---------------------------------------------------------------------------
# reset / unpatch
# ---------------------------------------------------------------------------

def reset_jano_state(self) -> None:
    """新一轮推理前调用，清空所有 Jano 运行时状态。"""
    _reset_jano_runtime(self)
    for _attr in ("_jano_step_level", "_jano_compute_mask_dev", "_jano_dynamic_idx"):
        if hasattr(self, _attr):
            delattr(self, _attr)
    if hasattr(self, "_jano_mm"):
        self._jano_mm.clear_cache()
    if hasattr(self, "_jano_da"):
        shape_key = getattr(self, "_jano_shape_key", None)
        if shape_key is not None:
            noise_h, noise_w, noise_seq_len, ref_seq_len = shape_key
            in_channels = getattr(self, "_jano_in_channels", 64)
            self._jano_da = DynamicAnalyzer(T=1, H=noise_h, W=noise_w, C=in_channels)
    if hasattr(self, "_jano_kv_cache"):
        self._jano_kv_cache.clear()


# ---------------------------------------------------------------------------
# Monkey patch
# ---------------------------------------------------------------------------

def apply_jano_monkey_patch() -> None:
    """将 Jano forward 绑定到 FireRed / diffusers / Cubic Transformer 类。"""
    from firered.transformer import QwenImageTransformer2DModel as FireRedModel

    FireRedModel.forward = jano_forward
    FireRedModel.reset_jano_state = reset_jano_state
    logger.info("[Jano] Patched FireRedQwenImageTransformer2DModel.forward")

    try:
        from diffusers.models import QwenImageTransformer2DModel as DiffusersModel

        if DiffusersModel is not FireRedModel:
            DiffusersModel.forward = jano_forward
            DiffusersModel.reset_jano_state = reset_jano_state
            logger.info("[Jano] Patched diffusers QwenImageTransformer2DModel.forward")
    except Exception as exc:
        logger.warning("[Jano] Skip diffusers patch: %s", exc)

    try:
        from firered_cubic.transformer import CubicQwenImageTransformer2DModel as CubicModel

        CubicModel.forward = jano_forward
        CubicModel.reset_jano_state = reset_jano_state
        logger.info("[Jano] Patched CubicQwenImageTransformer2DModel.forward")
    except Exception as exc:
        logger.warning("[Jano] Skip firered_cubic patch: %s", exc)


apply_jano_monkey_patch()
