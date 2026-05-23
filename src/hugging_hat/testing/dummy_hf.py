from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DummyConfig:
    hidden_size: int = 32
    num_hidden_layers: int = 8
    vocab_size: int = 128
    num_attention_heads: int = 4


class DummyBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.ff(self.norm(hidden_states))


class DummyCausalLM(nn.Module):
    """
    Minimal HF-like decoder-only CausalLM for local hook smoke tests.

    Exposes:
    - .config with hidden_size/num_hidden_layers/etc
    - .model.layers as the decoder block stack
    - forward(input_ids=..., attention_mask=..., past_key_values=...)
    - generate(...) that calls forward in a loop (not a real sampler)
    """

    def __init__(self, config: DummyConfig | None = None) -> None:
        super().__init__()
        self.config = config or DummyConfig()

        self.embed = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DummyBlock(self.config.hidden_size) for _ in range(self.config.num_hidden_layers)])
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        labels: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.embed(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        logits = self.lm_head(hidden_states)
        out: dict[str, torch.Tensor] = {"logits": logits, "hidden_states": hidden_states}
        if labels is not None:
            # Match HF causal-LM CE: shift so logits[..., t, :] predicts labels[..., t+1],
            # with ignore_index=-100. Returns nan when no valid positions remain.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            out["loss"] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 4,
        **kwargs: Any,
    ) -> torch.Tensor:
        tokens = input_ids
        past_key_values = None
        for _ in range(max_new_tokens):
            out = self.forward(
                input_ids=tokens[:, -1:],
                attention_mask=attention_mask[:, : tokens.shape[1]] if attention_mask is not None else None,
                past_key_values=past_key_values,
                **kwargs,
            )
            next_token = out["logits"][:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            past_key_values = object()
        return tokens


def build_dummy_causallm(*, hidden_size: int = 32, num_hidden_layers: int = 8) -> DummyCausalLM:
    return DummyCausalLM(DummyConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers))

