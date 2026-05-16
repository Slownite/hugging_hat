from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        variance = x_float.pow(2).mean(-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        return (x_norm * self.weight).to(dtype=x.dtype)

