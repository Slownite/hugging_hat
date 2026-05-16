from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .norm import RMSNorm


@dataclass(frozen=True, slots=True)
class ThinkerHatOutput:
    steps: int


class ThinkerHat(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        hidden_multiplier: int = 4,
        use_rms_norm: bool = True,
        compute_dtype: str = "match_base",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.compute_dtype = compute_dtype

        inner = hidden_size * hidden_multiplier
        self.norm = RMSNorm(hidden_size) if use_rms_norm else nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Linear(inner, hidden_size),
        )
        self.gate = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor, *, num_steps: int) -> tuple[torch.Tensor, ThinkerHatOutput]:
        if num_steps <= 0:
            return hidden_states, ThinkerHatOutput(steps=0)

        original_dtype = hidden_states.dtype
        if self.compute_dtype == "float32":
            hidden_states = hidden_states.float()

        for _ in range(num_steps):
            normalized = self.norm(hidden_states)
            delta = self.mlp(normalized)
            gate = torch.sigmoid(self.gate(normalized))  # (B,S,1)
            hidden_states = hidden_states + gate * delta

        hidden_states = hidden_states.to(dtype=original_dtype)
        return hidden_states, ThinkerHatOutput(steps=num_steps)

