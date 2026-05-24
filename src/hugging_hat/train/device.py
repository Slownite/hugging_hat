"""Device and mixed-precision resolution for v0 Thinker training.

See `docs/training-api-v0.md` (Boundaries → "Device + dtype/autocast policy")
and issue #9 for the contract this module implements.

Scope: choose a `torch.device`, build the autocast/GradScaler policy from a
`TrainConfig.precision` literal, and provide a single helper to move a batch
of tensors. Hat compute dtype (`HatConfig.*.compute_dtype`) is intentionally
*not* touched here — that knob is independent of the training-loop autocast
policy.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch


def resolve_device(requested: str | None) -> torch.device:
    """Pick a `torch.device`.

    - `None`  → cuda if available else cpu.
    - any other value is honored as-is (e.g. ``"cpu"``, ``"cuda"``, ``"cuda:1"``).
    """
    if requested is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


@dataclass(frozen=True)
class PrecisionPolicy:
    """Resolved autocast + GradScaler policy.

    When `autocast_device_type` is None, no autocast wrapping is performed
    and `scaler` is None. Otherwise the training step wraps the forward in
    `torch.autocast(autocast_device_type, dtype=autocast_dtype)` and uses
    `scaler` for the backward/step.
    """

    autocast_device_type: str | None
    autocast_dtype: torch.dtype | None
    scaler: "torch.amp.GradScaler | None"


def resolve_precision(
    precision: Literal["fp32", "fp16"],
    device: torch.device,
) -> PrecisionPolicy:
    """Build the autocast/scaler policy for ``precision`` on ``device``.

    fp16 on a non-cuda device emits a warning and degrades to fp32 rather than
    erroring: PyTorch CPU autocast targets bfloat16, and fp16 ``GradScaler`` is
    cuda-only.
    """
    if precision == "fp32":
        return PrecisionPolicy(None, None, None)
    if precision != "fp16":
        raise ValueError(f"Unsupported precision: {precision!r}")

    if device.type != "cuda":
        warnings.warn(
            "precision='fp16' is only supported on CUDA; "
            f"falling back to fp32 on device type {device.type!r}.",
            UserWarning,
            stacklevel=2,
        )
        return PrecisionPolicy(None, None, None)

    return PrecisionPolicy(
        autocast_device_type="cuda",
        autocast_dtype=torch.float16,
        scaler=torch.amp.GradScaler("cuda"),
    )


def move_batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move every tensor in `batch` to `device`. Non-tensor values pass through."""
    out: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out
