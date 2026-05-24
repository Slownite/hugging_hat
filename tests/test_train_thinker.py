"""Tests for the v0 Thinker-only training loop (issue #8).

These tests use ``DummyCausalLM`` from ``hugging_hat.testing.dummy_hf`` (which
now honors the HF labels→loss contract) so they run without network access.
"""

from __future__ import annotations

import torch

from hugging_hat.config import (
    HatConfig,
    LatentRouterHatConfig,
    ThinkerHatConfig,
)
from hugging_hat.data import PromptCompletion
from hugging_hat.model import HatEnabledModel
from hugging_hat.testing.dummy_hf import build_dummy_causallm
from hugging_hat.tokenizer import IGNORE_INDEX, preprocess_record
from hugging_hat.train import (
    StepMetrics,
    TrainConfig,
    collate,
    freeze_base_enable_hats,
    train_thinker,
    training_step,
)


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
    model = HatEnabledModel(base, config=_config())
    return model


def _hat_param_names(model: HatEnabledModel) -> set[str]:
    return {
        n for n, _ in model.named_parameters()
        if n.startswith(("thinker_hat.", "router_hat.", "critic_hat."))
    }


def _snapshot(model: HatEnabledModel) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


# ----------------------------------------------------------------------------- collate

def test_collate_pads_right_with_correct_values():
    pad_id = 7
    batch = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
    ]
    out = collate(batch, pad_token_id=pad_id)

    assert out["input_ids"].shape == (2, 3)
    assert out["attention_mask"].shape == (2, 3)
    assert out["labels"].shape == (2, 3)
    assert all(t.dtype is torch.long for t in out.values())

    assert out["input_ids"][1].tolist() == [4, 5, pad_id]
    assert out["attention_mask"][1].tolist() == [1, 1, 0]
    assert out["labels"][1].tolist() == [-100, 5, IGNORE_INDEX]


# ----------------------------------------------------------------------------- freeze

def test_freeze_base_enable_hats_partitions_grads():
    model = _build_model()
    hat_params = freeze_base_enable_hats(model)

    hat_names = _hat_param_names(model)
    assert hat_names, "fixture must enable at least one hat"

    for name, param in model.named_parameters():
        if name in hat_names:
            assert param.requires_grad, f"hat param {name} should require grad"
        else:
            assert not param.requires_grad, f"base param {name} must be frozen"

    returned_ids = {id(p) for p in hat_params}
    expected_ids = {id(p) for n, p in model.named_parameters() if n in hat_names}
    assert returned_ids == expected_ids


# ----------------------------------------------------------------------------- training_step

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
        labels[:, : seq_len // 2] = IGNORE_INDEX  # mask the "prompt" half
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def test_training_step_updates_hats_only_and_not_base():
    model = _build_model()
    hat_params = freeze_base_enable_hats(model)
    optimizer = torch.optim.AdamW(hat_params, lr=1e-2)

    before = _snapshot(model)
    batch = _make_batch()

    model.set_steps_override(4)
    try:
        metrics = training_step(
            model, batch, optimizer,
            thinker_steps=4,
            grad_clip=1.0,
            hat_params=hat_params,
        )
    finally:
        model.clear_steps_override()

    assert isinstance(metrics, StepMetrics)
    assert metrics.loss > 0.0

    hat_names = _hat_param_names(model)
    after = _snapshot(model)
    for name, param in model.named_parameters():
        if name in hat_names:
            assert not torch.equal(before[name], after[name]), (
                f"hat param {name} must change after a training step"
            )
        else:
            assert torch.equal(before[name], after[name]), (
                f"base param {name} must not change while frozen"
            )


def test_training_step_skips_when_all_labels_ignored():
    model = _build_model()
    hat_params = freeze_base_enable_hats(model)
    optimizer = torch.optim.AdamW(hat_params, lr=1e-2)

    before = _snapshot(model)
    batch = _make_batch(all_ignore=True)

    model.set_steps_override(4)
    try:
        metrics = training_step(
            model, batch, optimizer,
            thinker_steps=4,
            grad_clip=1.0,
            hat_params=hat_params,
        )
    finally:
        model.clear_steps_override()

    assert metrics.loss == 0.0
    after = _snapshot(model)
    for name, _ in model.named_parameters():
        assert torch.equal(before[name], after[name]), (
            f"param {name} must not change when no targets are valid"
        )


# ----------------------------------------------------------------------------- fake tokenizer + loop

class _FakeTokenizer:
    """Minimal tokenizer that satisfies preprocess_record's contract.

    Splits on whitespace, maps each token to ``hash(word) % vocab_size``, and
    prepends one BOS-like special token so the leading-special path in
    preprocess_record is exercised.
    """

    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = vocab_size
        self.bos_id = 1
        self.eos_id = 2
        self.pad_token_id: int | None = 0
        self.eos_token_id: int = self.eos_id

    def _tokens(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            # Reserve 0..2 for pad/bos/eos so content ids never collide.
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


def _overfit_records() -> list[PromptCompletion]:
    # A tiny dataset the dummy can overfit in a handful of steps.
    return [
        PromptCompletion(prompt="alpha beta", completion=" gamma delta"),
        PromptCompletion(prompt="hello", completion=" world"),
    ]


def test_train_thinker_loss_decreases_and_writes_checkpoint(tmp_path):
    tokenizer = _FakeTokenizer()
    records = _overfit_records()

    # Capture an "initial" loss by running one step on a clone of the setup.
    initial_model = _build_model()
    initial_hat_params = freeze_base_enable_hats(initial_model)
    initial_optimizer = torch.optim.AdamW(initial_hat_params, lr=1e-2)
    initial_model.set_steps_override(4)
    try:
        initial_batch = collate(
            [preprocess_record(r, tokenizer, max_length=32) for r in records],
            pad_token_id=int(tokenizer.pad_token_id or tokenizer.eos_token_id),
        )
        initial_metrics = training_step(
            initial_model, initial_batch, initial_optimizer,
            thinker_steps=4, hat_params=initial_hat_params,
        )
    finally:
        initial_model.clear_steps_override()

    # Now train a fresh model for several steps over the same data.
    model = _build_model()
    out_dir = tmp_path / "hats"
    config = TrainConfig(
        max_length=32,
        batch_size=2,
        lr=1e-2,
        weight_decay=0.0,
        num_epochs=20,
        grad_accum_steps=1,
        grad_clip=1.0,
        thinker_steps=4,
        log_every=0,  # silence prints
        save_every=None,
        device="cpu",
    )
    result = train_thinker(
        model, records, tokenizer, config, output_dir=str(out_dir),
    )

    assert result.steps >= 2
    assert result.final_loss < initial_metrics.loss, (
        f"loss should decrease on an overfit batch "
        f"(initial={initial_metrics.loss}, final={result.final_loss})"
    )
    assert (out_dir / "hats.safetensors").is_file()
    assert (out_dir / "hat_metadata.json").is_file()


def test_train_thinker_checkpoint_round_trips_via_load_hats(tmp_path):
    tokenizer = _FakeTokenizer()
    records = _overfit_records()
    out_dir = tmp_path / "hats"

    trained = _build_model(base_seed=0, hat_seed=1)
    train_thinker(
        trained, records, tokenizer,
        TrainConfig(
            max_length=32, batch_size=2, lr=1e-2,
            num_epochs=3, thinker_steps=2, log_every=0,
            device="cpu",
        ),
        output_dir=str(out_dir),
    )

    # Fresh model, differently-seeded hat init.
    fresh = _build_model(base_seed=0, hat_seed=9)

    trained_hat_state = {
        n: p.detach().clone()
        for n, p in trained.named_parameters()
        if n.startswith(("thinker_hat.", "router_hat.", "critic_hat."))
    }
    fresh_before = {
        n: p.detach().clone()
        for n, p in fresh.named_parameters()
        if n.startswith(("thinker_hat.", "router_hat.", "critic_hat."))
    }

    assert any(
        not torch.equal(trained_hat_state[n], fresh_before[n])
        for n in trained_hat_state
    ), "sanity: fresh hats must differ from trained hats before loading"

    fresh.load_hats(str(out_dir))

    for n, p in fresh.named_parameters():
        if n in trained_hat_state:
            assert torch.equal(p.detach(), trained_hat_state[n]), (
                f"loaded hat param {n} must match trained value"
            )
