"""v0 Thinker training loop.

See `docs/training-api-v0.md` for the design and per-symbol contracts.
"""

from __future__ import annotations

from .collate import collate
from .config import StepMetrics, TrainConfig, TrainResult
from .device import (
    PrecisionPolicy,
    move_batch_to_device,
    resolve_device,
    resolve_precision,
)
from .freeze import freeze_base_enable_hats
from .loop import train_thinker
from .step import training_step

__all__ = [
    "PrecisionPolicy",
    "StepMetrics",
    "TrainConfig",
    "TrainResult",
    "collate",
    "freeze_base_enable_hats",
    "move_batch_to_device",
    "resolve_device",
    "resolve_precision",
    "train_thinker",
    "training_step",
]
