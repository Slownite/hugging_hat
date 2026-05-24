"""Tests for resume_from semantics (issue #11, Option A: hat-weights only).

These tests pin the v0 contract: ``TrainConfig.resume_from`` loads hat weights
via ``model.load_hats`` and continues with a *fresh* optimizer and step counter.
Optimizer/scheduler/RNG state is intentionally NOT restored.
"""

from __future__ import annotations

import pytest
import torch

from hugging_hat.config import (
    HatConfig,
    LatentRouterHatConfig,
    ThinkerHatConfig,
)
from hugging_hat.model import HatEnabledModel
from hugging_hat.persistence import InvalidHatCheckpointError
from hugging_hat.testing.dummy_hf import build_dummy_causallm
from hugging_hat.train import TrainConfig, train_thinker

from .test_train_thinker import _FakeTokenizer, _overfit_records


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


def _hat_state(model: HatEnabledModel) -> dict[str, torch.Tensor]:
    return {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if n.startswith(("thinker_hat.", "router_hat.", "critic_hat."))
    }


def _train_config(**overrides) -> TrainConfig:
    base = dict(
        max_length=32,
        batch_size=2,
        lr=1e-2,
        num_epochs=3,
        thinker_steps=2,
        log_every=0,
        device="cpu",
    )
    base.update(overrides)
    return TrainConfig(**base)


def test_resume_from_loads_hat_weights_into_fresh_model(tmp_path):
    """A fresh model trained with resume_from=ckpt must start from ckpt weights.

    We assert this by setting num_epochs=0 so no training mutates weights; the
    only way the resumed model's weights can match the saved checkpoint is if
    ``resume_from`` actually called ``load_hats``.
    """
    tokenizer = _FakeTokenizer()
    records = _overfit_records()
    ckpt_dir = tmp_path / "hats"

    trained = _build_model(base_seed=0, hat_seed=1)
    train_thinker(
        trained, records, tokenizer,
        _train_config(num_epochs=3),
        output_dir=str(ckpt_dir),
    )
    saved_state = _hat_state(trained)

    # Fresh model with a *different* hat seed so its initial weights differ.
    fresh = _build_model(base_seed=0, hat_seed=9)
    fresh_initial = _hat_state(fresh)
    assert any(
        not torch.equal(saved_state[n], fresh_initial[n]) for n in saved_state
    ), "sanity: fresh hats must differ from trained hats before resume"

    # num_epochs=0 means: load weights, do not train, save and return.
    resume_out = tmp_path / "hats_after_resume"
    train_thinker(
        fresh, records, tokenizer,
        _train_config(num_epochs=0, resume_from=str(ckpt_dir)),
        output_dir=str(resume_out),
    )

    after_resume = _hat_state(fresh)
    for n in saved_state:
        assert torch.equal(after_resume[n], saved_state[n]), (
            f"hat param {n} must match checkpoint weights after resume"
        )


def test_resume_starts_fresh_step_counter(tmp_path):
    """Option A: resume does NOT carry the source run's step counter forward.

    With 2 records and batch_size=2, each epoch produces exactly one optimizer
    step. A 4-epoch source run reports steps=4. A 2-epoch resumed run must
    report steps=2 -- not 6 -- because the step counter is intentionally reset.
    """
    tokenizer = _FakeTokenizer()
    records = _overfit_records()
    ckpt_dir = tmp_path / "hats"

    source = _build_model()
    source_result = train_thinker(
        source, records, tokenizer,
        _train_config(num_epochs=4),
        output_dir=str(ckpt_dir),
    )
    assert source_result.steps == 4

    resumed = _build_model(hat_seed=9)
    resumed_result = train_thinker(
        resumed, records, tokenizer,
        _train_config(num_epochs=2, resume_from=str(ckpt_dir)),
        output_dir=str(tmp_path / "hats_resumed"),
    )

    assert resumed_result.steps == 2, (
        "Option A: resumed run's step counter must start at 0, "
        f"reporting only this run's steps; got {resumed_result.steps}"
    )


def test_resume_from_invalid_path_raises_invalid_checkpoint(tmp_path):
    """Pointing resume_from at a non-checkpoint dir surfaces a clear error."""
    tokenizer = _FakeTokenizer()
    records = _overfit_records()

    bogus = tmp_path / "not_a_checkpoint"
    bogus.mkdir()
    (bogus / "readme.txt").write_text("nothing useful here")

    model = _build_model()
    with pytest.raises(InvalidHatCheckpointError):
        train_thinker(
            model, records, tokenizer,
            _train_config(num_epochs=1, resume_from=str(bogus)),
            output_dir=str(tmp_path / "out"),
        )
