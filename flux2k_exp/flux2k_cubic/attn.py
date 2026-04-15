from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb

from flux2k_exp.flux2_klein.transformer import _get_qkv_projections


def _call_cache_update(
	engine,
	runtime: dict,
	layer_key: Tuple[str, int],
	key_active: torch.Tensor,
	value_active: torch.Tensor,
	active_idx: torch.Tensor,
	total_tokens: int,
	prefer_full: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
	cache_key = runtime["cache_key"]

	if prefer_full and hasattr(engine, "_update_and_get_full_kv_cache"):
		return engine._update_and_get_full_kv_cache(
			cache_key=cache_key,
			layer_key=layer_key,
			key_active=key_active,
			value_active=value_active,
			active_idx=active_idx,
			total_tokens=total_tokens,
		)

	if hasattr(engine, "_update_and_get_kv_cache"):
		return engine._update_and_get_kv_cache(
			cache_key=cache_key,
			layer_key=layer_key,
			key_active=key_active,
			value_active=value_active,
			active_idx=active_idx,
			total_tokens=total_tokens,
		)

	if not prefer_full and hasattr(engine, "_update_and_get_image_kv_cache"):
		return engine._update_and_get_image_kv_cache(
			cache_key=cache_key,
			layer_key=layer_key,
			key_active=key_active,
			value_active=value_active,
			active_idx=active_idx,
			total_tokens=total_tokens,
		)

	if prefer_full and hasattr(engine, "_update_and_get_image_kv_cache"):
		return engine._update_and_get_image_kv_cache(
			cache_key=cache_key,
			layer_key=layer_key,
			key_active=key_active,
			value_active=value_active,
			active_idx=active_idx,
			total_tokens=total_tokens,
		)

	raise AttributeError(
		"KV cache update method not found. Expected one of: "
		"_update_and_get_full_kv_cache, _update_and_get_kv_cache, _update_and_get_image_kv_cache"
	)


class Flux2CubicKVCacheAttnProcessor:
	"""Processor for Flux2Attention (double-stream block) with selective image KV cache.

	Rule alignment:
	1) keep encoder_hidden_states path complete;
	2) only image hidden_states KV is updated/completed from cache;
	3) KV update/merge is done after RoPE.
	"""

	_attention_backend = None
	_parallel_config = None

	def __init__(self, engine, layer_kind: str, layer_idx: int):
		if not hasattr(F, "scaled_dot_product_attention"):
			raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")

		self.engine = engine
		self.layer_kind = layer_kind
		self.layer_idx = layer_idx
		print(f"Initialized {self.__class__.__name__} for layer ({layer_kind}, {layer_idx})", flush=True)

	def __call__(
		self,
		attn,
		hidden_states: torch.Tensor,
		encoder_hidden_states: Optional[torch.Tensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		image_rotary_emb: Optional[torch.Tensor] = None,
	):
		query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
			attn, hidden_states, encoder_hidden_states
		)

		query = query.unflatten(-1, (attn.heads, -1))
		key = key.unflatten(-1, (attn.heads, -1))
		value = value.unflatten(-1, (attn.heads, -1))

		query = attn.norm_q(query)
		key = attn.norm_k(key)

		if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
			encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
			encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
			encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

			encoder_query = attn.norm_added_q(encoder_query)
			encoder_key = attn.norm_added_k(encoder_key)

			query = torch.cat([encoder_query, query], dim=1)
			key = torch.cat([encoder_key, key], dim=1)
			value = torch.cat([encoder_value, value], dim=1)

		if image_rotary_emb is not None:
			query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
			key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

		runtime = getattr(self.engine, "_runtime", None)
		if runtime is not None and encoder_hidden_states is not None:
			text_len = int(encoder_hidden_states.shape[1])
			key_text, key_img_active = key[:, :text_len, :, :], key[:, text_len:, :, :]
			value_text, value_img_active = value[:, :text_len, :, :], value[:, text_len:, :, :]

			key_img_full, value_img_full = _call_cache_update(
				engine=self.engine,
				runtime=runtime,
				layer_key=(self.layer_kind, self.layer_idx),
				key_active=key_img_active,
				value_active=value_img_active,
				active_idx=runtime["active_idx"],
				total_tokens=int(runtime["total_tokens"]),
				prefer_full=False,
			)

			key = torch.cat([key_text, key_img_full], dim=1)
			value = torch.cat([value_text, value_img_full], dim=1)
			
		print(f"attn {self.layer_idx}: {query.shape=}, {key.shape=}, {value.shape=}", flush=True)
		hidden_states = dispatch_attention_fn(
			query,
			key,
			value,
			attn_mask=attention_mask,
			backend=self._attention_backend,
			parallel_config=self._parallel_config,
		)
		hidden_states = hidden_states.flatten(2, 3)
		hidden_states = hidden_states.to(query.dtype)

		if encoder_hidden_states is not None:
			encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
				[encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
			)
			encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

		hidden_states = attn.to_out[0](hidden_states)
		hidden_states = attn.to_out[1](hidden_states)

		if encoder_hidden_states is not None:
			return hidden_states, encoder_hidden_states
		return hidden_states


class Flux2CubicKVCacheParallelSelfAttnProcessor:
	"""Processor for Flux2ParallelSelfAttention with whole-sequence selective KV cache.

	Rule alignment:
	1) no encoder_hidden_states branch;
	2) update/complete frozen KV on the whole concatenated sequence;
	3) KV update/merge is done after RoPE.
	"""

	_attention_backend = None
	_parallel_config = None

	def __init__(self, engine, layer_kind: str, layer_idx: int):
		if not hasattr(F, "scaled_dot_product_attention"):
			raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")

		self.engine = engine
		self.layer_kind = layer_kind
		self.layer_idx = layer_idx
		print(f"Initialized {self.__class__.__name__} for layer ({layer_kind}, {layer_idx})", flush=True)

	def __call__(
		self,
		attn,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		image_rotary_emb: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		projected = attn.to_qkv_mlp_proj(hidden_states)
		qkv, mlp_hidden_states = torch.split(
			projected, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
		)

		query, key, value = qkv.chunk(3, dim=-1)

		query = query.unflatten(-1, (attn.heads, -1))
		key = key.unflatten(-1, (attn.heads, -1))
		value = value.unflatten(-1, (attn.heads, -1))

		query = attn.norm_q(query)
		key = attn.norm_k(key)

		if image_rotary_emb is not None:
			query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
			key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

		runtime = getattr(self.engine, "_runtime", None)
		if runtime is not None:
			encoder_tokens = int(runtime.get("encoder_tokens", 0))
			img_active_idx = runtime["active_idx"]
			img_total_tokens = int(runtime["total_tokens"])

			if encoder_tokens > 0:
				text_idx = torch.arange(encoder_tokens, device=img_active_idx.device, dtype=img_active_idx.dtype)
				seq_active_idx = torch.cat([text_idx, img_active_idx + encoder_tokens], dim=0)
			else:
				seq_active_idx = img_active_idx

			seq_total_tokens = encoder_tokens + img_total_tokens
			key, value = _call_cache_update(
				engine=self.engine,
				runtime=runtime,
				layer_key=(self.layer_kind, self.layer_idx),
				key_active=key,
				value_active=value,
				active_idx=seq_active_idx,
				total_tokens=seq_total_tokens,
				prefer_full=True,
			)

		attn_output = dispatch_attention_fn(
			query,
			key,
			value,
			attn_mask=attention_mask,
			backend=self._attention_backend,
			parallel_config=self._parallel_config,
		)
		attn_output = attn_output.flatten(2, 3)
		attn_output = attn_output.to(query.dtype)

		mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
		hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
		hidden_states = attn.to_out(hidden_states)
		return hidden_states

