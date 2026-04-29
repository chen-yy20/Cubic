"""
ToCa (Token Cache) adaptation for FireRed / Qwen-Image Transformer.

This version fixes the two major issues in the previous draft:
1. CFG branches are really isolated: each branch owns independent caches.
2. MLP cache stores MLP *outputs*, not norm/modulated MLP inputs.

Core policy, following the practical ToCa implementation:
- Full step: run the original block, cache attention outputs and MLP outputs.
- ToCa step: reuse the whole attention output; selectively recompute MLP for fresh tokens.

Usage:
    import baseline.tocache_firered_fixed  # monkey patch on import

    transformer = pipe.transformer
    transformer.enable_tocache = True
    transformer.num_steps = num_inference_steps
    transformer.tocache_fresh_ratio = 0.10
    transformer.tocache_warmup_steps = 3
    transformer.tocache_cal_threshold = 4
    transformer.reset_tocache_state()
"""

from __future__ import annotations

import functools
import weakref
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import apply_lora_scale, deprecate, logging

from firered.transformer import compute_text_seq_len_from_mask

logger = logging.get_logger(__name__)


# -----------------------------------------------------------------------------
# Cache/runtime data structures
# -----------------------------------------------------------------------------

def _make_branch_cache(num_layers: int) -> dict[str, Any]:
    """Create one independent cache bank, e.g. for CFG branch_0 or branch_1."""
    return {
        "cache": {layer: {} for layer in range(num_layers)},
        "cache_index": {layer: {} for layer in range(num_layers)},
        "score": {layer: {} for layer in range(num_layers)},
        "layer_ready": [False for _ in range(num_layers)],
        "ready": False,
    }


def _init_tocache_cache(num_layers: int) -> dict[str, Any]:
    """Initialize ToCa cache container. Branch caches are created lazily."""
    return {
        "num_layers": int(num_layers),
        "branches": {},
        # default hyper-parameters; updated from transformer attributes each forward
        "fresh_ratio": 0.10,
        "warmup_steps": 3,
        "cal_threshold": 4,
        "layer_ratio_slope": 0.50,       # deeper layers get larger refresh ratio
        "force_fresh": "global",
        "soft_fresh_weight": 0.25,
        "current_norm_weight": 0.05,
        "min_fresh_tokens": 1,
    }


def _get_branch_cache(cache_dic: dict[str, Any], branch_key: str) -> dict[str, Any]:
    branches = cache_dic.setdefault("branches", {})
    if branch_key not in branches:
        branches[branch_key] = _make_branch_cache(cache_dic["num_layers"])
    return branches[branch_key]


def _make_tocache_runtime() -> dict[str, Any]:
    return {
        "last_timestep": None,
        "step_index": 0,
        "call_index_for_step": -1,
    }


def _reset_tocache_runtime(self) -> None:
    self._tocache_runtime = _make_tocache_runtime()
    self._tocache_cache = _init_tocache_cache(len(self.transformer_blocks))
    self._tocache_current = {
        "step": 0,
        "layer": 0,
        "module": "img_mlp",
        "type": "full",
        "branch": "branch_0",
        "num_steps": int(getattr(self, "num_steps", 50)),
        "final_time": int(getattr(self, "num_steps", 50)) - 1,
    }


def _ensure_tocache_runtime(self) -> None:
    if not hasattr(self, "_tocache_runtime") or not hasattr(self, "_tocache_cache"):
        _reset_tocache_runtime(self)


def reset_tocache_state(self) -> None:
    """Call before each new generation to clear all ToCa runtime states and caches."""
    _reset_tocache_runtime(self)


def _update_tocache_runtime(self, timestep: torch.Tensor, num_steps: int) -> tuple[int, str]:
    """
    Infer denoising step index and CFG branch from call order.

    In common CFG-by-two-forward pipelines, the same timestep is called twice:
    branch_0 and branch_1 get independent cache banks.
    """
    _ensure_tocache_runtime(self)

    current_timestep = float(timestep.detach().flatten()[0].float().cpu().item())
    runtime = self._tocache_runtime
    last_timestep = runtime["last_timestep"]
    same_step = last_timestep is not None and abs(current_timestep - last_timestep) <= 1e-12

    if last_timestep is None:
        runtime["step_index"] = 0
        runtime["call_index_for_step"] = 0
    elif same_step:
        runtime["call_index_for_step"] += 1
    else:
        order = str(getattr(self, "tocache_timestep_order", "descending"))
        if order == "descending":
            new_generation = current_timestep > last_timestep
        elif order == "ascending":
            new_generation = current_timestep < last_timestep
        else:
            new_generation = False

        # Also reset if the previous generation has already consumed all expected steps.
        new_generation = new_generation or runtime["step_index"] >= max(int(num_steps) - 1, 0)

        if new_generation:
            _reset_tocache_runtime(self)
            runtime = self._tocache_runtime
            runtime["step_index"] = 0
            runtime["call_index_for_step"] = 0
        else:
            runtime["step_index"] += 1
            runtime["call_index_for_step"] = 0

    runtime["last_timestep"] = current_timestep
    branch_key = f"branch_{min(int(runtime['call_index_for_step']), 1)}"
    return int(runtime["step_index"]), branch_key


# -----------------------------------------------------------------------------
# Scheduling and token selection
# -----------------------------------------------------------------------------

def _cal_type(cache_dic: dict[str, Any], current: dict[str, Any], branch_cache: dict[str, Any]) -> str:
    """Return 'full' or 'ToCa' for the current denoising step."""
    step = int(current["step"])
    warmup_steps = int(cache_dic.get("warmup_steps", 3))
    interval = max(int(cache_dic.get("cal_threshold", 4)), 1)

    # Force full until this branch has a valid cache for all layers.
    if not bool(branch_cache.get("ready", False)):
        current["type"] = "full"
        return "full"

    if step < warmup_steps:
        current["type"] = "full"
    else:
        # With warmup=3 and interval=4: steps 0,1,2 full; 3,4,5 ToCa; 6 full; ...
        current["type"] = "full" if ((step - warmup_steps + 1) % interval == 0) else "ToCa"

    return current["type"]


def _fresh_ratio_scheduler(cache_dic: dict[str, Any], current: dict[str, Any]) -> float:
    """Layer-aware fresh-token ratio. Deeper layers receive a larger refresh ratio."""
    base_ratio = float(cache_dic.get("fresh_ratio", 0.10))
    layer = int(current.get("layer", 0))
    num_layers = max(int(cache_dic.get("num_layers", 60)), 1)
    slope = float(cache_dic.get("layer_ratio_slope", 0.50))

    if num_layers == 1:
        layer_pos = 0.5
    else:
        layer_pos = layer / float(num_layers - 1)  # shallow=0, deep=1

    layer_factor = 1.0 + slope * (layer_pos - 0.5)
    return float(np.clip(base_ratio * layer_factor, 0.0, 1.0))


def _normalize_score(score: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max normalization for stable ranking."""
    score = score.float()
    score = score - score.amin(dim=1, keepdim=True)
    denom = score.amax(dim=1, keepdim=True).clamp_min(1e-6)
    return score / denom


def _ensure_cache_index(
    branch_cache: dict[str, Any],
    layer: int,
    module: str,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Return stale-counter tensor [B, L], creating/resetting it if the shape changed."""
    index_bank = branch_cache["cache_index"][layer]
    expected_shape = tokens.shape[:2]
    index = index_bank.get(module)
    if index is None or tuple(index.shape) != tuple(expected_shape) or index.device != tokens.device:
        index = torch.zeros(expected_shape, dtype=torch.long, device=tokens.device)
        index_bank[module] = index
    return index


def _score_evaluate(
    cache_dic: dict[str, Any],
    branch_cache: dict[str, Any],
    current: dict[str, Any],
    tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Score tokens for fresh computation.

    Higher score -> higher priority to recompute.
    This FireRed adaptation uses cached attention/MLP-output norm as a robust proxy,
    and adds stale-counter forcing so no token stays stale forever.
    """
    layer = int(current["layer"])
    module = str(current["module"])

    score_bank = branch_cache["score"][layer]
    score = score_bank.get(module)
    if score is None or tuple(score.shape) != tuple(tokens.shape[:2]) or score.device != tokens.device:
        score = tokens.detach().norm(dim=-1)
    score = _normalize_score(score.to(device=tokens.device))

    current_norm_weight = float(cache_dic.get("current_norm_weight", 0.05))
    if current_norm_weight > 0:
        score = score + current_norm_weight * _normalize_score(tokens.detach().norm(dim=-1))

    if cache_dic.get("force_fresh", "global") == "global":
        index = _ensure_cache_index(branch_cache, layer, module, tokens)
        stale = index.float() / float(max(int(cache_dic.get("cal_threshold", 4)), 1))
        score = score + float(cache_dic.get("soft_fresh_weight", 0.25)) * _normalize_score(stale)

    return score.to(device=tokens.device)


def _cache_cutfresh(
    cache_dic: dict[str, Any],
    branch_cache: dict[str, Any],
    tokens: torch.Tensor,
    current: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select fresh tokens and update stale counters."""
    layer = int(current["layer"])
    module = str(current["module"])
    batch, seq_len = tokens.shape[:2]

    ratio = _fresh_ratio_scheduler(cache_dic, current)
    topk = int(round(ratio * seq_len))
    min_fresh = int(cache_dic.get("min_fresh_tokens", 1))
    if ratio > 0 and seq_len > 0:
        topk = max(min_fresh, topk)
    topk = min(max(topk, 0), seq_len)

    cache_index = _ensure_cache_index(branch_cache, layer, module, tokens)
    cache_index.add_(1)

    if topk == 0:
        empty_idx = torch.empty((batch, 0), dtype=torch.long, device=tokens.device)
        empty_tokens = tokens[:, :0, :]
        return empty_idx, empty_tokens

    score = _score_evaluate(cache_dic, branch_cache, current, tokens)
    fresh_indices = torch.topk(score, k=topk, dim=1, largest=True, sorted=False).indices

    cache_index.scatter_(
        dim=1,
        index=fresh_indices,
        src=torch.zeros_like(fresh_indices, dtype=cache_index.dtype, device=tokens.device),
    )

    gather_index = fresh_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    fresh_tokens = torch.gather(tokens, dim=1, index=gather_index)
    return fresh_indices, fresh_tokens


def _update_token_cache(
    branch_cache: dict[str, Any],
    layer: int,
    module: str,
    fresh_indices: torch.Tensor,
    fresh_values: torch.Tensor,
) -> None:
    """Scatter fresh MLP outputs back into the full cached MLP-output tensor."""
    if fresh_indices.numel() == 0:
        return

    full_cache = branch_cache["cache"][layer].get(module)
    if full_cache is None:
        raise RuntimeError(f"Missing cache for layer={layer}, module={module}")

    if full_cache.shape[0] != fresh_values.shape[0] or full_cache.shape[-1] != fresh_values.shape[-1]:
        raise RuntimeError(
            f"Cache shape mismatch for {module}: cache={tuple(full_cache.shape)}, "
            f"fresh={tuple(fresh_values.shape)}"
        )

    scatter_index = fresh_indices.unsqueeze(-1).expand(-1, -1, fresh_values.shape[-1])
    full_cache.scatter_(dim=1, index=scatter_index, src=fresh_values.detach())

    # Keep score cache synchronized for future token selection.
    score = branch_cache["score"][layer].get(module)
    if score is not None and tuple(score.shape) == tuple(full_cache.shape[:2]):
        fresh_score = fresh_values.detach().norm(dim=-1).to(score.device)
        score.scatter_(dim=1, index=fresh_indices.to(score.device), src=fresh_score)


def _store_full_tensor_cache(
    branch_cache: dict[str, Any],
    layer: int,
    module: str,
    tensor: torch.Tensor,
    score_tensor: Optional[torch.Tensor] = None,
) -> None:
    """Store a full layer tensor and initialize/update its score and stale counter."""
    branch_cache["cache"][layer][module] = tensor.detach().clone()

    if score_tensor is None:
        score_tensor = tensor.detach().norm(dim=-1)
    branch_cache["score"][layer][module] = score_tensor.detach().clone()

    branch_cache["cache_index"][layer][module] = torch.zeros(
        tensor.shape[:2], dtype=torch.long, device=tensor.device
    )


def _clip_fp16(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.float16:
        return x.clip(-65504, 65504)
    return x


# -----------------------------------------------------------------------------
# Block-level patch
# -----------------------------------------------------------------------------

def _patch_tocache_blocks(transformer) -> None:
    """Patch each block instance. Full mode uses original block.forward; ToCa mode is manual."""
    transformer_ref = weakref.ref(transformer)
    num_blocks = len(transformer.transformer_blocks)

    for block_idx, block in enumerate(transformer.transformer_blocks):
        # ------------------------------------------------------------------
        # 1) Attention patch: in full mode, capture attention outputs.
        # ------------------------------------------------------------------
        if not hasattr(block.attn, "_tocache_original_forward"):
            block.attn._tocache_original_forward = block.attn.forward
            orig_attn_forward = block.attn._tocache_original_forward

            def _make_attn_forward(orig_fn, blk_idx, tr_ref):
                @functools.wraps(orig_fn)
                def tocache_attn_forward(*args, **kwargs):
                    tr = tr_ref()
                    if tr is None or not getattr(tr, "enable_tocache", False):
                        return orig_fn(*args, **kwargs)

                    output = orig_fn(*args, **kwargs)

                    current = getattr(tr, "_tocache_current", {})
                    if current.get("type") == "full":
                        try:
                            cache_dic = tr._tocache_cache
                            branch_cache = _get_branch_cache(cache_dic, current.get("branch", "branch_0"))

                            if isinstance(output, tuple):
                                img_attn_output = output[0]
                                txt_attn_output = output[1] if len(output) > 1 else None
                            else:
                                img_attn_output = output
                                txt_attn_output = None

                            branch_cache["cache"][blk_idx]["img_attn"] = img_attn_output.detach().clone()
                            branch_cache["score"][blk_idx]["img_mlp"] = img_attn_output.detach().norm(dim=-1)

                            if txt_attn_output is not None:
                                branch_cache["cache"][blk_idx]["txt_attn"] = txt_attn_output.detach().clone()
                                branch_cache["score"][blk_idx]["txt_mlp"] = txt_attn_output.detach().norm(dim=-1)
                        except Exception as exc:  # keep original computation safe
                            logger.warning("[ToCa] failed to cache attention at block %d: %s", blk_idx, exc)

                    return output

                return tocache_attn_forward

            block.attn.forward = _make_attn_forward(orig_attn_forward, block_idx, transformer_ref)

        # ------------------------------------------------------------------
        # 2) Block patch: full mode delegates to original; ToCa mode reuses cache.
        # ------------------------------------------------------------------
        if hasattr(block, "_tocache_original_forward"):
            continue

        block._tocache_original_forward = block.forward
        orig_block_forward = block._tocache_original_forward

        def _make_block_forward(orig_fn, blk_idx, tr_ref):
            @functools.wraps(orig_fn)
            def tocache_block_forward(
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                temb,
                image_rotary_emb=None,
                joint_attention_kwargs=None,
                modulate_index=None,
            ):
                tr = tr_ref()
                if tr is None or not getattr(tr, "enable_tocache", False):
                    return orig_fn(
                        hidden_states,
                        encoder_hidden_states,
                        encoder_hidden_states_mask,
                        temb,
                        image_rotary_emb,
                        joint_attention_kwargs,
                        modulate_index,
                    )

                cache_dic = tr._tocache_cache
                current = tr._tocache_current
                branch_cache = _get_branch_cache(cache_dic, current.get("branch", "branch_0"))
                block_obj = tr.transformer_blocks[blk_idx]
                calc_type = current.get("type", "full")

                # ----------------------------------------------------------
                # FULL: use original block output, then reconstruct and cache
                # attention/MLP outputs for future ToCa steps.
                # ----------------------------------------------------------
                if calc_type == "full":
                    result = orig_fn(
                        hidden_states,
                        encoder_hidden_states,
                        encoder_hidden_states_mask,
                        temb,
                        image_rotary_emb,
                        joint_attention_kwargs,
                        modulate_index,
                    )

                    try:
                        with torch.no_grad():
                            img_mod1, img_mod2 = block_obj.img_mod(temb).chunk(2, dim=-1)
                            txt_mod1, txt_mod2 = block_obj.txt_mod(temb).chunk(2, dim=-1)

                            img_attn = branch_cache["cache"][blk_idx].get("img_attn")
                            txt_attn = branch_cache["cache"][blk_idx].get("txt_attn")
                            if img_attn is None:
                                raise RuntimeError("attention output was not captured")

                            img_normed = block_obj.img_norm1(hidden_states)
                            _, img_gate1 = block_obj._modulate(img_normed, img_mod1, modulate_index)
                            post_img_attn = hidden_states + img_gate1 * img_attn

                            txt_normed = block_obj.txt_norm1(encoder_hidden_states)
                            _, txt_gate1 = block_obj._modulate(txt_normed, txt_mod1)
                            post_txt_attn = encoder_hidden_states
                            if txt_attn is not None:
                                post_txt_attn = encoder_hidden_states + txt_gate1 * txt_attn

                            img_normed2 = block_obj.img_norm2(post_img_attn)
                            img_modulated2, _ = block_obj._modulate(img_normed2, img_mod2, modulate_index)
                            img_mlp_out = block_obj.img_mlp(img_modulated2)

                            img_score = img_mlp_out.detach().norm(dim=-1)
                            if "img_mlp" in branch_cache["score"][blk_idx]:
                                img_score = img_score + branch_cache["score"][blk_idx]["img_mlp"].to(img_score.device)
                            _store_full_tensor_cache(branch_cache, blk_idx, "img_mlp", img_mlp_out, img_score)

                            if hasattr(block_obj, "txt_mlp"):
                                txt_normed2 = block_obj.txt_norm2(post_txt_attn)
                                txt_modulated2, _ = block_obj._modulate(txt_normed2, txt_mod2)
                                txt_mlp_out = block_obj.txt_mlp(txt_modulated2)
                                txt_score = txt_mlp_out.detach().norm(dim=-1)
                                if "txt_mlp" in branch_cache["score"][blk_idx]:
                                    txt_score = txt_score + branch_cache["score"][blk_idx]["txt_mlp"].to(txt_score.device)
                                _store_full_tensor_cache(branch_cache, blk_idx, "txt_mlp", txt_mlp_out, txt_score)

                            # Make sure attention cache is shape-consistent for this layer.
                            if img_attn.shape[:2] != hidden_states.shape[:2]:
                                raise RuntimeError("image attention cache shape mismatch")
                            branch_cache["layer_ready"][blk_idx] = True
                    except Exception as exc:
                        branch_cache["layer_ready"][blk_idx] = False
                        logger.warning("[ToCa] full-cache update failed at block %d: %s", blk_idx, exc)

                    return result

                # ----------------------------------------------------------
                # ToCa: reuse full attention outputs; selectively refresh MLP.
                # ----------------------------------------------------------
                if calc_type == "ToCa":
                    try:
                        layer_cache = branch_cache["cache"][blk_idx]
                        img_attn = layer_cache["img_attn"]
                        img_mlp_cache = layer_cache["img_mlp"]
                        txt_attn = layer_cache.get("txt_attn")
                        txt_mlp_cache = layer_cache.get("txt_mlp")

                        if img_attn.shape[:2] != hidden_states.shape[:2]:
                            raise RuntimeError(
                                f"image attention cache shape mismatch: {tuple(img_attn.shape)} vs {tuple(hidden_states.shape)}"
                            )
                        if img_mlp_cache.shape[:2] != hidden_states.shape[:2]:
                            raise RuntimeError(
                                f"image MLP cache shape mismatch: {tuple(img_mlp_cache.shape)} vs {tuple(hidden_states.shape)}"
                            )

                        img_mod1, img_mod2 = block_obj.img_mod(temb).chunk(2, dim=-1)
                        txt_mod1, txt_mod2 = block_obj.txt_mod(temb).chunk(2, dim=-1)

                        # Attention residual: whole attention output is reused.
                        img_normed = block_obj.img_norm1(hidden_states)
                        _, img_gate1 = block_obj._modulate(img_normed, img_mod1, modulate_index)
                        hidden_states = hidden_states + img_gate1 * img_attn

                        txt_normed = block_obj.txt_norm1(encoder_hidden_states)
                        _, txt_gate1 = block_obj._modulate(txt_normed, txt_mod1)
                        if txt_attn is not None:
                            if txt_attn.shape[:2] != encoder_hidden_states.shape[:2]:
                                raise RuntimeError("text attention cache shape mismatch")
                            encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn

                        # Image MLP selective refresh.
                        current["layer"] = blk_idx
                        current["module"] = "img_mlp"
                        img_normed2 = block_obj.img_norm2(hidden_states)
                        img_modulated2, img_gate2 = block_obj._modulate(img_normed2, img_mod2, modulate_index)

                        fresh_idx, fresh_tokens = _cache_cutfresh(cache_dic, branch_cache, img_modulated2, current)
                        if fresh_idx.numel() > 0:
                            fresh_mlp = block_obj.img_mlp(fresh_tokens)
                            _update_token_cache(branch_cache, blk_idx, "img_mlp", fresh_idx, fresh_mlp)

                        img_mlp_out = branch_cache["cache"][blk_idx]["img_mlp"]
                        hidden_states = hidden_states + img_gate2 * img_mlp_out

                        # Text MLP selective refresh, if text MLP cache exists.
                        if txt_mlp_cache is not None and hasattr(block_obj, "txt_mlp"):
                            current["module"] = "txt_mlp"
                            txt_normed2 = block_obj.txt_norm2(encoder_hidden_states)
                            txt_modulated2, txt_gate2 = block_obj._modulate(txt_normed2, txt_mod2)

                            fresh_idx_txt, fresh_tokens_txt = _cache_cutfresh(
                                cache_dic, branch_cache, txt_modulated2, current
                            )
                            if fresh_idx_txt.numel() > 0:
                                fresh_txt_mlp = block_obj.txt_mlp(fresh_tokens_txt)
                                _update_token_cache(branch_cache, blk_idx, "txt_mlp", fresh_idx_txt, fresh_txt_mlp)

                            txt_mlp_out = branch_cache["cache"][blk_idx]["txt_mlp"]
                            encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_out

                        encoder_hidden_states = _clip_fp16(encoder_hidden_states)
                        hidden_states = _clip_fp16(hidden_states)
                        return encoder_hidden_states, hidden_states

                    except Exception as exc:
                        logger.warning("[ToCa] block %d ToCa failed; fallback to original full block: %s", blk_idx, exc)
                        # Fallback is safe but does not update cache for this block.
                        return orig_fn(
                            hidden_states,
                            encoder_hidden_states,
                            encoder_hidden_states_mask,
                            temb,
                            image_rotary_emb,
                            joint_attention_kwargs,
                            modulate_index,
                        )

                return orig_fn(
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                    modulate_index,
                )

            return tocache_block_forward

        block.forward = _make_block_forward(orig_block_forward, block_idx, transformer_ref)

    logger.info("[ToCa] patched %d transformer blocks on %s", num_blocks, type(transformer).__name__)


def _unpatch_tocache_blocks(transformer) -> None:
    """Restore original block/attention forwards for debugging."""
    for block in transformer.transformer_blocks:
        if hasattr(block.attn, "_tocache_original_forward"):
            block.attn.forward = block.attn._tocache_original_forward
            del block.attn._tocache_original_forward
        if hasattr(block, "_tocache_original_forward"):
            block.forward = block._tocache_original_forward
            del block._tocache_original_forward
    transformer._tocache_blocks_patched = False
    logger.info("[ToCa] unpatched transformer blocks on %s", type(transformer).__name__)


# -----------------------------------------------------------------------------
# Transformer-level forward
# -----------------------------------------------------------------------------

@apply_lora_scale("attention_kwargs")
def tocache_forward(
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
    ToCa-aware forward for FireRed QwenImageTransformer2DModel and Cubic variant.

    The non-ToCa path intentionally mirrors firered/transformer.py style.
    """
    if txt_seq_lens is not None:
        deprecate(
            "txt_seq_lens",
            "0.39.0",
            "Passing `txt_seq_lens` is deprecated and will be removed in version 0.39.0. "
            "Please use `encoder_hidden_states_mask` instead.",
            standard_warn=False,
        )

    enable_tocache = bool(getattr(self, "enable_tocache", False))
    num_steps = int(getattr(self, "num_steps", 50))

    # ------------------------------------------------------------------
    # 1. ToCa runtime update
    # ------------------------------------------------------------------
    if enable_tocache:
        _ensure_tocache_runtime(self)
        step_index, branch_key = _update_tocache_runtime(self, timestep, num_steps)

        cache_dic = self._tocache_cache
        cache_dic["fresh_ratio"] = float(getattr(self, "tocache_fresh_ratio", 0.10))
        cache_dic["warmup_steps"] = int(getattr(self, "tocache_warmup_steps", 3))
        cache_dic["cal_threshold"] = int(getattr(self, "tocache_cal_threshold", 4))
        cache_dic["layer_ratio_slope"] = float(getattr(self, "tocache_layer_ratio_slope", 0.50))
        cache_dic["soft_fresh_weight"] = float(getattr(self, "tocache_soft_fresh_weight", 0.25))
        cache_dic["current_norm_weight"] = float(getattr(self, "tocache_current_norm_weight", 0.05))
        cache_dic["min_fresh_tokens"] = int(getattr(self, "tocache_min_fresh_tokens", 1))

        self._tocache_current.update(
            {
                "step": step_index,
                "num_steps": num_steps,
                "final_time": num_steps - 1,
                "branch": branch_key,
            }
        )

        if not getattr(self, "_tocache_blocks_patched", False):
            _patch_tocache_blocks(self)
            self._tocache_blocks_patched = True

        branch_cache = _get_branch_cache(cache_dic, branch_key)
        _cal_type(cache_dic, self._tocache_current, branch_cache)

    # ------------------------------------------------------------------
    # 2. Standard FireRed transformer pre-processing
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3. Transformer blocks. Patched blocks decide full vs ToCa internally.
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

    # Mark this branch ready only after all layer caches were successfully created.
    if enable_tocache:
        branch_cache = _get_branch_cache(self._tocache_cache, self._tocache_current.get("branch", "branch_0"))
        branch_cache["ready"] = bool(all(branch_cache["layer_ready"]))

    # ------------------------------------------------------------------
    # 4. Output projection
    # ------------------------------------------------------------------
    if self.zero_cond_t:
        temb = temb.chunk(2, dim=0)[0]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


# -----------------------------------------------------------------------------
# Monkey patch entry
# -----------------------------------------------------------------------------

def apply_tocache_monkey_patch() -> None:
    """Bind ToCa forward to FireRed / diffusers / Cubic Transformer classes."""
    from firered.transformer import QwenImageTransformer2DModel as FireRedQwenImageTransformer2DModel

    if not hasattr(FireRedQwenImageTransformer2DModel, "_tocache_original_class_forward"):
        FireRedQwenImageTransformer2DModel._tocache_original_class_forward = FireRedQwenImageTransformer2DModel.forward
    FireRedQwenImageTransformer2DModel.forward = tocache_forward
    FireRedQwenImageTransformer2DModel.reset_tocache_state = reset_tocache_state
    FireRedQwenImageTransformer2DModel.unpatch_tocache_blocks = _unpatch_tocache_blocks

    try:
        from diffusers.models import QwenImageTransformer2DModel as DiffusersQwenImageTransformer2DModel

        if DiffusersQwenImageTransformer2DModel is not FireRedQwenImageTransformer2DModel:
            if not hasattr(DiffusersQwenImageTransformer2DModel, "_tocache_original_class_forward"):
                DiffusersQwenImageTransformer2DModel._tocache_original_class_forward = DiffusersQwenImageTransformer2DModel.forward
            DiffusersQwenImageTransformer2DModel.forward = tocache_forward
            DiffusersQwenImageTransformer2DModel.reset_tocache_state = reset_tocache_state
            DiffusersQwenImageTransformer2DModel.unpatch_tocache_blocks = _unpatch_tocache_blocks
    except Exception as exc:
        logger.warning("[ToCa] skip diffusers QwenImageTransformer2DModel patch: %s", exc)

    try:
        from firered_cubic.transformer import CubicQwenImageTransformer2DModel

        if not hasattr(CubicQwenImageTransformer2DModel, "_tocache_original_class_forward"):
            CubicQwenImageTransformer2DModel._tocache_original_class_forward = CubicQwenImageTransformer2DModel.forward
        CubicQwenImageTransformer2DModel.forward = tocache_forward
        CubicQwenImageTransformer2DModel.reset_tocache_state = reset_tocache_state
        CubicQwenImageTransformer2DModel.unpatch_tocache_blocks = _unpatch_tocache_blocks
    except Exception as exc:
        logger.warning("[ToCa] skip firered_cubic patch: %s", exc)


apply_tocache_monkey_patch()
