import logging
from typing import Dict, Optional, Tuple

import torch

from wan.modules.attention import flash_attention
from wan.modules.model import sinusoidal_embedding_1d


class WanSelectiveForwardEngine:
    """Selective sparse forward for WAN T2V using StepPlan.

    This engine keeps origin Wan modules unchanged by running a wrapper forward:
    - active tokens: full block compute (self-attn + cross-attn + ffn)
    - frozen tokens: self-attn K/V are reused from cache
    - frozen hidden states reuse cached pre-head residuals (TeaCache-style)
    """

    def __init__(self, config):
        self.config = config
        self.k_cache: Dict[Tuple[str, int], torch.Tensor] = {}
        self.v_cache: Dict[Tuple[str, int], torch.Tensor] = {}
        self.residual_cache: Dict[str, torch.Tensor] = {}

    def reset(self):
        self.k_cache.clear()
        self.v_cache.clear()
        self.residual_cache.clear()

    def forward(
        self,
        model,
        x,
        t,
        context,
        seq_len,
        branch_key: str,
        step_plan=None,
        clip_fea=None,
        y=None,
    ):
        if model.model_type != 't2v':
            return model(x, t=t, context=context, seq_len=seq_len, clip_fea=clip_fea, y=y)

        if len(x) != 1:
            raise ValueError('Selective forward currently supports batch size 1 only for WAN T2V.')

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        device = model.patch_embedding.weight.device
        if model.freqs.device != device:
            model.freqs = model.freqs.to(device)

        # Embedding path mirrors WanModel.forward, but keeps valid sequence length only.
        x_emb = [model.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x_emb])
        x_seq = [u.flatten(2).transpose(1, 2) for u in x_emb]
        seq_lens = torch.tensor([u.size(1) for u in x_seq], dtype=torch.long)
        x_full = x_seq[0]
        total_tokens = int(seq_lens[0].item())

        with torch.cuda.amp.autocast(dtype=torch.float32):
            e = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, t).float())
            e0 = model.time_projection(e).unflatten(1, (6, model.dim))

        context_lens = None
        context = model.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(model.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )

        if clip_fea is not None:
            context_clip = model.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        # Decide full vs selective path.
        use_selective = (
            step_plan is not None
            and (not step_plan.is_full_compute)
            and step_plan.active_token_indices.numel() > 0
        )

        if use_selective:
            active_idx = step_plan.active_token_indices.long().to(device)
            if int(active_idx.numel()) >= total_tokens:
                use_selective = False

        if not use_selective:
            out = self._full_forward_and_refresh_cache(
                model=model,
                x_full=x_full,
                e=e,
                e0=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                context=context,
                context_lens=context_lens,
                branch_key=branch_key,
            )
            return out

        # Selective path requires existing cache for frozen tokens.
        if not self._has_cache(model, branch_key):
            logging.info(
                '[wan-selective] branch=%s missing cache; fallback to full forward for cache bootstrap.',
                branch_key,
            )
            return self._full_forward_and_refresh_cache(
                model=model,
                x_full=x_full,
                e=e,
                e0=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                context=context,
                context_lens=context_lens,
                branch_key=branch_key,
            )

        logging.info(
            '[wan-selective] branch=%s active_tokens=%d frozen_tokens=%d',
            branch_key,
            int(step_plan.active_token_indices.numel()),
            int(step_plan.frozen_token_indices.numel()),
        )

        x_active = x_full.index_select(1, active_idx)

        for layer_idx, block in enumerate(model.blocks):
            x_active = self._selective_block_forward(
                block=block,
                x_active=x_active,
                e_time=e0,
                active_idx=active_idx,
                total_tokens=total_tokens,
                grid_size=grid_sizes[0],
                freqs=model.freqs,
                context=context,
                context_lens=context_lens,
                branch_key=branch_key,
                layer_idx=layer_idx,
            )

        residual_cache = self.residual_cache.get(branch_key)
        if residual_cache is None or residual_cache.size(0) != total_tokens:
            logging.info(
                '[wan-selective] branch=%s residual cache invalid; fallback to full forward.',
                branch_key,
            )
            return self._full_forward_and_refresh_cache(
                model=model,
                x_full=x_full,
                e=e,
                e0=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                context=context,
                context_lens=context_lens,
                branch_key=branch_key,
            )

        residual_cache = residual_cache.to(x_full.device)
        hidden_full = x_full[0] + residual_cache
        hidden_full[active_idx] = x_active[0]

        updated_residual = residual_cache.clone()
        updated_residual[active_idx] = x_active[0] - x_full[0, active_idx]
        self.residual_cache[branch_key] = updated_residual.detach()

        out_full = model.head(hidden_full.unsqueeze(0), e)[0]
        denoised = model.unpatchify(out_full.unsqueeze(0), grid_sizes)
        return [u.float() for u in denoised]

    def _full_forward_and_refresh_cache(
        self,
        model,
        x_full,
        e,
        e0,
        seq_lens,
        grid_sizes,
        context,
        context_lens,
        branch_key: str,
    ):
        total_tokens = int(seq_lens[0].item())
        all_idx = torch.arange(total_tokens, device=x_full.device, dtype=torch.long)
        x_active = x_full

        for layer_idx, block in enumerate(model.blocks):
            x_active = self._selective_block_forward(
                block=block,
                x_active=x_active,
                e_time=e0,
                active_idx=all_idx,
                total_tokens=total_tokens,
                grid_size=grid_sizes[0],
                freqs=model.freqs,
                context=context,
                context_lens=context_lens,
                branch_key=branch_key,
                layer_idx=layer_idx,
            )

        residual_full = x_active[0] - x_full[0]
        self.residual_cache[branch_key] = residual_full.detach()
        out_full = model.head(x_active, e)[0]

        denoised = model.unpatchify(out_full.unsqueeze(0), grid_sizes)
        return [u.float() for u in denoised]

    def _selective_block_forward(
        self,
        block,
        x_active,
        e_time,
        active_idx,
        total_tokens: int,
        grid_size,
        freqs,
        context,
        context_lens,
        branch_key: str,
        layer_idx: int,
    ):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            e0, e1, e2, e3, e4, e5 = (block.modulation + e_time).chunk(6, dim=1)

        # self-attention: Q from active, K/V from active + cached frozen.
        x_for_attn = block.norm1(x_active).float() * (1 + e1) + e0
        bsz, active_len, _ = x_for_attn.shape
        num_heads = block.self_attn.num_heads
        head_dim = block.self_attn.head_dim

        q = block.self_attn.norm_q(block.self_attn.q(x_for_attn)).view(bsz, active_len, num_heads, head_dim)
        k_active = block.self_attn.norm_k(block.self_attn.k(x_for_attn)).view(bsz, active_len, num_heads, head_dim)
        v_active = block.self_attn.v(x_for_attn).view(bsz, active_len, num_heads, head_dim)

        q = self._rope_apply_indexed(q, active_idx, grid_size, freqs)
        k_active = self._rope_apply_indexed(k_active, active_idx, grid_size, freqs)

        k_key = (branch_key, layer_idx)
        v_key = (branch_key, layer_idx)

        k_cache = self.k_cache.get(k_key)
        v_cache = self.v_cache.get(v_key)
        if k_cache is None or v_cache is None or k_cache.size(0) != total_tokens:
            k_cache = q.new_zeros((total_tokens, num_heads, head_dim))
            v_cache = v_active.new_zeros((total_tokens, num_heads, head_dim))

        k_cache = k_cache.to(q.device)
        v_cache = v_cache.to(v_active.device)
        k_cache[active_idx] = k_active[0]
        v_cache[active_idx] = v_active[0]
        self.k_cache[k_key] = k_cache.detach()
        self.v_cache[v_key] = v_cache.detach()

        k_full = k_cache.unsqueeze(0)
        v_full = v_cache.unsqueeze(0)

        q_lens = torch.tensor([active_len], dtype=torch.long)
        k_lens = torch.tensor([total_tokens], dtype=torch.long)

        y = flash_attention(
            q=q,
            k=k_full,
            v=v_full,
            q_lens=q_lens,
            k_lens=k_lens,
            window_size=block.self_attn.window_size,
        )
        y = block.self_attn.o(y.flatten(2))

        with torch.cuda.amp.autocast(dtype=torch.float32):
            x_active = x_active + y * e2

        # cross-attention + ffn on active tokens only.
        x_active = x_active + block.cross_attn(block.norm3(x_active), context, context_lens)
        y_ffn = block.ffn(block.norm2(x_active).float() * (1 + e4) + e3)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            x_active = x_active + y_ffn * e5

        logging.debug(
            '[wan-kv-cache] branch=%s layer=%d active=%d total=%d',
            branch_key,
            layer_idx,
            int(active_len),
            int(total_tokens),
        )
        return x_active

    def _has_cache(self, model, branch_key: str) -> bool:
        if branch_key not in self.residual_cache:
            return False
        for layer_idx in range(len(model.blocks)):
            if (branch_key, layer_idx) not in self.k_cache:
                return False
            if (branch_key, layer_idx) not in self.v_cache:
                return False
        return True

    def _rope_apply_indexed(self, x, token_indices, grid_size, freqs):
        # x shape: [1, L_active, num_heads, head_dim]
        _, _, n_heads, head_dim = x.shape
        half = head_dim // 2

        f, h, w = [int(v) for v in grid_size.tolist()]
        hw = h * w

        idx = token_indices.long()
        f_idx = torch.div(idx, hw, rounding_mode='floor')
        rem = idx - f_idx * hw
        h_idx = torch.div(rem, w, rounding_mode='floor')
        w_idx = rem - h_idx * w

        freqs_split = freqs.split([half - 2 * (half // 3), half // 3, half // 3], dim=1)
        f_part = freqs_split[0].to(x.device).index_select(0, f_idx)
        h_part = freqs_split[1].to(x.device).index_select(0, h_idx)
        w_part = freqs_split[2].to(x.device).index_select(0, w_idx)
        freq_token = torch.cat([f_part, h_part, w_part], dim=1).unsqueeze(1)

        x_complex = torch.view_as_complex(x[0].to(torch.float64).reshape(x.size(1), n_heads, -1, 2))
        out = torch.view_as_real(x_complex * freq_token).flatten(2).unsqueeze(0)
        return out.float()
