from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class LatentRouterOutput:
    chosen_steps: torch.Tensor  # (B,)
    logits: torch.Tensor  # (B, K)


class LatentRouterHat(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        step_set: tuple[int, ...] = (0, 2, 4, 8),
        compute_dtype: str = "match_base",
    ) -> None:
        super().__init__()
        if len(step_set) == 0:
            raise ValueError("step_set must be non-empty")
        if len(set(step_set)) != len(step_set):
            raise ValueError("step_set must not contain duplicates")

        self.hidden_size = hidden_size
        self.step_set = step_set
        self.compute_dtype = compute_dtype

        self.classifier = nn.Linear(hidden_size, len(step_set))

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
    ) -> LatentRouterOutput:
        original_dtype = hidden_states.dtype
        if self.compute_dtype == "float32":
            hidden_states = hidden_states.float()

        pooled = self._mean_pool(hidden_states, attention_mask=attention_mask)
        logits = self.classifier(pooled)
        chosen = logits.argmax(dim=-1)
        step_values = torch.tensor(self.step_set, device=logits.device, dtype=torch.long)
        chosen_steps = step_values.index_select(0, chosen)

        return LatentRouterOutput(
            chosen_steps=chosen_steps,
            logits=logits,
        )

    @staticmethod
    def _mean_pool(hidden_states: torch.Tensor, *, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)  # (B,S,1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_states * mask).sum(dim=1) / denom
