from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TrainConfig:
    # data
    max_length: int = 1024
    batch_size: int = 1
    # optim
    lr: float = 1e-4
    weight_decay: float = 0.0
    max_steps: int | None = None
    num_epochs: int = 1
    grad_accum_steps: int = 1
    grad_clip: float | None = 1.0
    # thinker training
    thinker_steps: int = 4
    # runtime
    device: str | None = None
    precision: Literal["fp32", "fp16"] = "fp32"
    seed: int | None = None
    # logging
    log_every: int = 10
    # checkpointing
    save_every: int | None = None
    resume_from: str | None = None


@dataclass(frozen=True)
class StepMetrics:
    step: int
    loss: float


@dataclass
class TrainResult:
    steps: int
    final_loss: float
    checkpoint_path: str | None
