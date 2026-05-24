from __future__ import annotations

import contextlib
from typing import Any

import torch
from torch import nn

from .config import StepMetrics


def _extract_loss(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "loss") and outputs.loss is not None:
        return outputs.loss
    if isinstance(outputs, dict) and outputs.get("loss") is not None:
        return outputs["loss"]
    raise RuntimeError(
        "Model forward did not return a 'loss'. Ensure 'labels' is in the batch "
        "and the base model computes CE when labels are provided."
    )


def training_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    scaler: torch.amp.GradScaler | None = None,
    autocast_device_type: str | None = None,
    autocast_dtype: torch.dtype | None = None,
    thinker_steps: int,
    grad_clip: float | None = None,
    step_index: int = 0,
    hat_params: list[nn.Parameter] | None = None,
) -> StepMetrics:
    """One forward → backward → (optional clip) → optimizer step → zero_grad.

    ``thinker_steps`` is accepted for API symmetry; the loop is expected to have
    already called ``model.set_steps_override(thinker_steps)`` once and to clear
    it on exit. We do not toggle it per-step here.

    If the batch has no valid (non-ignored) target positions the loss is non-
    finite; we skip the backward/step and return loss=0.0. This keeps the
    "all-(-100) batch → no gradient to hats" contract from issue #8.

    When ``autocast_device_type`` is set the forward runs inside
    ``torch.autocast(autocast_device_type, dtype=autocast_dtype)``; pair this
    with a cuda ``GradScaler`` for fp16 (issue #9, ``resolve_precision``).
    """
    del thinker_steps  # set by the loop, not by us

    model.train()
    if autocast_device_type is not None:
        autocast_ctx: contextlib.AbstractContextManager = torch.autocast(
            device_type=autocast_device_type, dtype=autocast_dtype
        )
    else:
        autocast_ctx = contextlib.nullcontext()

    with autocast_ctx:
        outputs = model(**batch)
        loss = _extract_loss(outputs)

    if not torch.isfinite(loss):
        return StepMetrics(step=step_index, loss=0.0)

    if scaler is not None:
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(hat_params or [], grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(hat_params or [], grad_clip)
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    return StepMetrics(step=step_index, loss=float(loss.detach().item()))
