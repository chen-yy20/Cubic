from __future__ import annotations

from math import prod
from typing import Any

import numpy as np
import torch

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import apply_lora_scale, deprecate

from firered.transformer import (
    QwenImageTransformer2DModel as FireRedQwenImageTransformer2DModel,
    compute_text_seq_len_from_mask,
)


class CubicQwenImageTransformer2DModel(FireRedQwenImageTransformer2DModel):
    """FireRed transformer with optional image-token sub-selection for cubic selective steps."""

    @staticmethod
    def _normalize_image_token_indices(
        image_token_indices: torch.Tensor | None,
        expected_length: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if image_token_indices is None:
            return None

        indices = torch.unique(image_token_indices.to(device=device, dtype=torch.long), sorted=True)
        if int(indices.numel()) != int(expected_length):
            raise ValueError(
                "image_token_indices length must match hidden_states sequence length "
                f"({int(indices.numel())} vs {int(expected_length)})."
            )
        return indices

    @staticmethod
    def _slice_image_rotary_emb(
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        image_token_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_token_indices is None:
            return image_rotary_emb

        img_freqs, txt_freqs = image_rotary_emb
        img_freqs = img_freqs.index_select(0, image_token_indices.to(img_freqs.device))
        return img_freqs, txt_freqs

    @apply_lora_scale("attention_kwargs")
    def forward(
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
    ) -> torch.Tensor | Transformer2DModelOutput:
        if txt_seq_lens is not None:
            deprecate(
                "txt_seq_lens",
                "0.39.0",
                "Passing `txt_seq_lens` is deprecated and will be removed in version 0.39.0. "
                "Please use `encoder_hidden_states_mask` instead. "
                "The mask-based approach is more flexible and supports variable-length sequences.",
                standard_warn=False,
            )

        hidden_states = self.img_in(hidden_states)
        image_token_indices = self._normalize_image_token_indices(
            image_token_indices=image_token_indices,
            expected_length=hidden_states.shape[1],
            device=hidden_states.device,
        )

        timestep = timestep.to(hidden_states.dtype)

        if self.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [[0] * prod(sample[0]) + [1] * sum(prod(s) for s in sample[1:]) for sample in img_shapes],
                device=timestep.device,
                dtype=torch.int,
            )
            if image_token_indices is not None:
                modulate_index = modulate_index.index_select(dim=1, index=image_token_indices.to(modulate_index.device))
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
        image_rotary_emb = self._slice_image_rotary_emb(image_rotary_emb, image_token_indices)

        block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        if encoder_hidden_states_mask is not None:
            batch_size, image_seq_len = hidden_states.shape[:2]
            image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
            joint_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
            block_attention_kwargs["attention_mask"] = joint_attention_mask

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

        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
