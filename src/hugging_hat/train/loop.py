from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import AdamW

from hugging_hat.data import PromptCompletion
from hugging_hat.model import HatEnabledModel
from hugging_hat.tokenizer import preprocess_record

from .collate import collate
from .config import StepMetrics, TrainConfig, TrainResult
from .freeze import freeze_base_enable_hats
from .step import training_step


def _resolve_pad_token_id(tokenizer: Any) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        return int(pad_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        raise ValueError(
            "Tokenizer has neither pad_token_id nor eos_token_id; cannot pad batches."
        )
    warnings.warn(
        "Tokenizer has no pad_token_id; falling back to eos_token_id "
        f"({int(eos_id)}) for padding.",
        UserWarning,
        stacklevel=2,
    )
    return int(eos_id)


def _iter_batches(
    records: Iterable[PromptCompletion],
    tokenizer: Any,
    *,
    max_length: int,
    batch_size: int,
    pad_token_id: int,
) -> Iterable[dict[str, torch.Tensor]]:
    buffer: list[dict[str, list[int]]] = []
    for record in records:
        buffer.append(preprocess_record(record, tokenizer, max_length=max_length))
        if len(buffer) == batch_size:
            yield collate(buffer, pad_token_id=pad_token_id)
            buffer = []
    if buffer:
        yield collate(buffer, pad_token_id=pad_token_id)


def _set_seed(seed: int | None) -> None:
    # Minimal v0 placeholder; full implementation lands with issue #10.
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_thinker(
    model: HatEnabledModel,
    records: Iterable[PromptCompletion],
    tokenizer: Any,
    config: TrainConfig,
    *,
    output_dir: str,
) -> TrainResult:
    _set_seed(config.seed)

    if config.resume_from is not None:
        model.load_hats(config.resume_from)

    hat_params = freeze_base_enable_hats(model)
    optimizer = AdamW(hat_params, lr=config.lr, weight_decay=config.weight_decay)
    pad_token_id = _resolve_pad_token_id(tokenizer)

    records_list = list(records)

    optim_step = 0
    last_loss = 0.0
    model.set_steps_override(config.thinker_steps)
    try:
        for epoch in range(config.num_epochs):
            if config.max_steps is not None and optim_step >= config.max_steps:
                break

            accum_loss_sum = 0.0
            accum_count = 0
            for batch in _iter_batches(
                records_list,
                tokenizer,
                max_length=config.max_length,
                batch_size=config.batch_size,
                pad_token_id=pad_token_id,
            ):
                if config.grad_accum_steps == 1:
                    metrics = training_step(
                        model,
                        batch,
                        optimizer,
                        thinker_steps=config.thinker_steps,
                        grad_clip=config.grad_clip,
                        step_index=optim_step,
                        hat_params=hat_params,
                    )
                    last_loss = metrics.loss
                    optim_step += 1
                else:
                    metrics = _accumulate_microbatch(
                        model, batch, optimizer, hat_params,
                        accum_count=accum_count,
                        grad_accum_steps=config.grad_accum_steps,
                        grad_clip=config.grad_clip,
                    )
                    accum_loss_sum += metrics.loss
                    accum_count += 1
                    if accum_count == config.grad_accum_steps:
                        last_loss = accum_loss_sum / accum_count
                        accum_loss_sum = 0.0
                        accum_count = 0
                        optim_step += 1
                    else:
                        continue  # don't tick step / log / save mid-accumulation

                if config.log_every > 0 and optim_step % config.log_every == 0:
                    print(f"[train_thinker] epoch={epoch} step={optim_step} loss={last_loss:.4f}")

                if config.save_every is not None and optim_step % config.save_every == 0:
                    model.save_hats(output_dir)

                if config.max_steps is not None and optim_step >= config.max_steps:
                    break
    finally:
        model.clear_steps_override()

    model.save_hats(output_dir)
    return TrainResult(steps=optim_step, final_loss=last_loss, checkpoint_path=output_dir)


def _accumulate_microbatch(
    model: HatEnabledModel,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    hat_params: list[torch.nn.Parameter],
    *,
    accum_count: int,
    grad_accum_steps: int,
    grad_clip: float | None,
) -> StepMetrics:
    """One micro-step of gradient accumulation.

    Each call performs forward + scaled backward. On the boundary
    (``accum_count == grad_accum_steps - 1``) the optimizer step,
    clip, and zero_grad happen.
    """
    model.train()
    outputs = model(**batch)
    if hasattr(outputs, "loss") and outputs.loss is not None:
        loss = outputs.loss
    elif isinstance(outputs, dict) and outputs.get("loss") is not None:
        loss = outputs["loss"]
    else:
        raise RuntimeError("Model forward did not return a 'loss'.")

    is_boundary = accum_count == grad_accum_steps - 1
    if torch.isfinite(loss):
        (loss / grad_accum_steps).backward()
    if is_boundary:
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(hat_params, grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    loss_value = float(loss.detach().item()) if torch.isfinite(loss) else 0.0
    return StepMetrics(step=accum_count, loss=loss_value)
