"""Tests for v0 logging + determinism in the Thinker training loop (issue #10).

Covers:
- ``StepMetrics.num_tokens`` reports count of non-ignored label positions.
- ``train_thinker`` with the same seed produces identical loss trajectories.
- ``train_thinker`` emits a periodic log line with step / loss / tokens/sec /
  thinker_steps to stdout when ``log_every > 0``.
"""

from __future__ import annotations

import re
from typing import Any

import torch

from hugging_hat.config import (
    HatConfig,
    LatentRouterHatConfig,
    ThinkerHatConfig,
)
from hugging_hat.data import PromptCompletion
from hugging_hat.model import HatEnabledModel
from hugging_hat.testing.dummy_hf import build_dummy_causallm
from hugging_hat.tokenizer import IGNORE_INDEX
from hugging_hat.train import (
    TrainConfig,
    freeze_base_enable_hats,
    train_thinker,
    training_step,
)


# ----------------------------------------------------------------------------- helpers

def _config() -> HatConfig:
    return HatConfig(
        layers_path="model.layers",
        thinker=ThinkerHatConfig(enabled=True, default_steps=0),
        router=LatentRouterHatConfig(enabled=False),
    )


def _build_model(*, base_seed: int = 0, hat_seed: int = 1) -> HatEnabledModel:
    torch.manual_seed(base_seed)
    base = build_dummy_causallm(hidden_size=32, num_hidden_layers=4)
    torch.manual_seed(hat_seed)
    return HatEnabledModel(base, config=_config())


class _FakeTokenizer:
    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = vocab_size
        self.bos_id = 1
        self.eos_id = 2
        self.pad_token_id: int | None = 0
        self.eos_token_id: int = self.eos_id

    def _tokens(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            ids.append(3 + (abs(hash(word)) % (self.vocab_size - 3)))
        return ids

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = False,
        max_length: int | None = None,
        add_special_tokens: bool = True,
        return_special_tokens_mask: bool = False,
    ) -> dict[str, list[int]]:
        ids = self._tokens(text)
        if add_special_tokens:
            ids = [self.bos_id, *ids]
            special_mask = [1] + [0] * (len(ids) - 1)
        else:
            special_mask = [0] * len(ids)
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            special_mask = special_mask[:max_length]
        out: dict[str, list[int]] = {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }
        if return_special_tokens_mask:
            out["special_tokens_mask"] = special_mask
        return out


def _records() -> list[PromptCompletion]:
    return [
        PromptCompletion(prompt="alpha beta", completion=" gamma delta"),
        PromptCompletion(prompt="hello", completion=" world"),
    ]


def _make_batch(
    *, batch_size: int = 2, seq_len: int = 8, vocab: int = 128, all_ignore: bool = False
) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    if all_ignore:
        labels = torch.full_like(input_ids, IGNORE_INDEX)
    else:
        labels = input_ids.clone()
        labels[:, : seq_len // 2] = IGNORE_INDEX
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ----------------------------------------------------------------------------- num_tokens

def test_step_metrics_reports_num_tokens_for_partial_mask():
    model = _build_model()
    hat_params = freeze_base_enable_hats(model)
    optimizer = torch.optim.AdamW(hat_params, lr=1e-2)

    batch = _make_batch(batch_size=2, seq_len=8)
    expected = int((batch["labels"] != IGNORE_INDEX).sum().item())
    assert expected > 0

    model.set_steps_override(4)
    try:
        metrics = training_step(
            model, batch, optimizer,
            thinker_steps=4, grad_clip=1.0, hat_params=hat_params,
        )
    finally:
        model.clear_steps_override()
    assert metrics.num_tokens == expected


def test_step_metrics_num_tokens_zero_when_all_ignored():
    model = _build_model()
    hat_params = freeze_base_enable_hats(model)
    optimizer = torch.optim.AdamW(hat_params, lr=1e-2)

    batch = _make_batch(all_ignore=True)

    model.set_steps_override(4)
    try:
        metrics = training_step(
            model, batch, optimizer,
            thinker_steps=4, grad_clip=1.0, hat_params=hat_params,
        )
    finally:
        model.clear_steps_override()
    assert metrics.num_tokens == 0
    assert metrics.loss == 0.0


# ----------------------------------------------------------------------------- determinism

def _train_and_collect_losses(
    seed: int, *, log_every: int = 1, num_epochs: int = 3,
    tmp_path: Any,
) -> list[float]:
    """Run a deterministic training session and parse losses from stdout."""
    model = _build_model()
    config = TrainConfig(
        max_length=32, batch_size=2, lr=1e-2,
        num_epochs=num_epochs, thinker_steps=2,
        log_every=log_every, save_every=None,
        device="cpu", seed=seed,
    )
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train_thinker(
            model, _records(), _FakeTokenizer(), config,
            output_dir=str(tmp_path),
        )
    # Lines look like "[train_thinker] ... loss=<float> ..."
    losses: list[float] = []
    for line in buf.getvalue().splitlines():
        m = re.search(r"loss=([0-9.eE+\-]+)", line)
        if m:
            losses.append(float(m.group(1)))
    return losses


def test_same_seed_yields_identical_loss_trajectory(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    out_a.mkdir()
    out_b.mkdir()
    losses_a = _train_and_collect_losses(seed=0, tmp_path=out_a)
    losses_b = _train_and_collect_losses(seed=0, tmp_path=out_b)
    assert losses_a, "expected at least one logged loss"
    assert losses_a == losses_b


# ----------------------------------------------------------------------------- logging

def test_train_thinker_logs_step_loss_throughput(tmp_path, capsys):
    model = _build_model()
    config = TrainConfig(
        max_length=32, batch_size=2, lr=1e-2,
        num_epochs=2, thinker_steps=2,
        log_every=1, save_every=None,
        device="cpu", seed=0,
    )
    train_thinker(
        model, _records(), _FakeTokenizer(), config,
        output_dir=str(tmp_path),
    )
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("[train_thinker]")]
    assert lines, f"no log lines found in stdout:\n{out!r}"

    sample = lines[0]
    # Required fields per issue #10 acceptance criteria.
    assert re.search(r"step=\d+", sample), sample
    assert re.search(r"loss=[0-9.eE+\-]+", sample), sample
    assert re.search(r"tokens/sec=[0-9.eE+\-]+", sample), sample
    assert re.search(r"thinker_steps=\d+", sample), sample
