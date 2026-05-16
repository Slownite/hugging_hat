from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class CrossAttentiveCriticOutput:
    gate_mean: torch.Tensor  # scalar


class CrossAttentiveCriticHat(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        compute_dtype: str = "match_base",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.compute_dtype = compute_dtype

        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        prompt_memory: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, CrossAttentiveCriticOutput]:
        original_dtype = hidden_states.dtype
        if self.compute_dtype == "float32":
            hidden_states = hidden_states.float()
            prompt_memory = prompt_memory.float()

        query = self.norm(hidden_states)
        key_value = prompt_memory

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.to(dtype=torch.bool)

        attended, _ = self.attn(query=query, key=key_value, value=key_value, key_padding_mask=key_padding_mask)
        gate = torch.sigmoid(self.gate(query))  # (B,S,1)
        updated = hidden_states + gate * attended
        updated = updated.to(dtype=original_dtype)
        return updated, CrossAttentiveCriticOutput(gate_mean=gate.mean())

