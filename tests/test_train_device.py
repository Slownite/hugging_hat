"""Tests for issue #9: device + mixed-precision policy."""

from __future__ import annotations

import warnings

import pytest
import torch

from hugging_hat.config import (
    HatConfig,
    LatentRouterHatConfig,
    ThinkerHatConfig,
)
from hugging_hat.data import PromptCompletion
from hugging_hat.model import HatEnabledModel
from hugging_hat.testing.dummy_hf import build_dummy_causallm
from hugging_hat.train import (
    PrecisionPolicy,
    TrainConfig,
    move_batch_to_device,
    resolve_device,
    resolve_precision,
    train_thinker,
)


# ----------------------------------------------------------------------------- resolve_device

def test_resolve_device_none_returns_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(None) == torch.device("cpu")


def test_resolve_device_none_returns_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(None) == torch.device("cuda")


def test_resolve_device_honors_explicit_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_honors_explicit_cuda_string(monkeypatch):
    # "cuda" should be honored as-is regardless of availability; the actual
    # .to(device) is what would fail loudly if the requested device is missing.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda") == torch.device("cuda")


# ----------------------------------------------------------------------------- resolve_precision

def test_resolve_precision_fp32_returns_noop_policy():
    policy = resolve_precision("fp32", torch.device("cpu"))
    assert policy == PrecisionPolicy(None, None, None)


def test_resolve_precision_fp16_on_cpu_warns_and_degrades():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        policy = resolve_precision("fp16", torch.device("cpu"))

    assert policy.autocast_device_type is None
    assert policy.autocast_dtype is None
    assert policy.scaler is None
    assert any(
        issubclass(w.category, UserWarning) and "fp16" in str(w.message)
        for w in caught
    ), f"expected fp16-on-cpu UserWarning, got {[str(w.message) for w in caught]}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_resolve_precision_fp16_on_cuda_builds_scaler():
    policy = resolve_precision("fp16", torch.device("cuda"))
    assert policy.autocast_device_type == "cuda"
    assert policy.autocast_dtype == torch.float16
    assert policy.scaler is not None


def test_resolve_precision_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported precision"):
        resolve_precision("bf16", torch.device("cpu"))  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- move_batch_to_device

def test_move_batch_to_device_moves_tensors_and_preserves_others():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "meta": "ignored",
    }
    moved = move_batch_to_device(batch, torch.device("cpu"))
    assert moved["input_ids"].device == torch.device("cpu")
    assert moved["attention_mask"].device == torch.device("cpu")
    assert moved["meta"] == "ignored"


# ----------------------------------------------------------------------------- end-to-end CPU smoke

class _FakeTokenizer:
    """Same minimal shim used in test_train_thinker."""

    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = vocab_size
        self.bos_id = 1
        self.eos_id = 2
        self.pad_token_id: int | None = 0
        self.eos_token_id: int = self.eos_id

    def _tokens(self, text: str) -> list[int]:
        return [3 + (abs(hash(word)) % (self.vocab_size - 3)) for word in text.split()]

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


def _build_model() -> HatEnabledModel:
    torch.manual_seed(0)
    base = build_dummy_causallm(hidden_size=32, num_hidden_layers=4)
    torch.manual_seed(1)
    return HatEnabledModel(
        base,
        config=HatConfig(
            layers_path="model.layers",
            thinker=ThinkerHatConfig(enabled=True, default_steps=0),
            router=LatentRouterHatConfig(enabled=False),
        ),
    )


def _hat_param_names(model: HatEnabledModel) -> set[str]:
    return {
        n for n, _ in model.named_parameters()
        if n.startswith(("thinker_hat.", "router_hat.", "critic_hat."))
    }


def test_train_thinker_cpu_smoke_with_explicit_device(tmp_path):
    model = _build_model()
    records = [
        PromptCompletion(prompt="alpha beta", completion=" gamma delta"),
        PromptCompletion(prompt="hello", completion=" world"),
    ]
    hat_names = _hat_param_names(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if n in hat_names}

    result = train_thinker(
        model,
        records,
        _FakeTokenizer(),
        TrainConfig(
            max_length=32,
            batch_size=2,
            lr=1e-2,
            num_epochs=2,
            thinker_steps=2,
            log_every=0,
            device="cpu",
            precision="fp32",
        ),
        output_dir=str(tmp_path / "hats"),
    )

    assert result.steps >= 1
    assert next(model.parameters()).device == torch.device("cpu")
    for n, p in model.named_parameters():
        if n in hat_names:
            assert not torch.equal(before[n], p.detach()), (
                f"hat param {n} must change after a CPU training run"
            )


def test_train_thinker_fp16_on_cpu_warns_and_completes(tmp_path):
    """fp16 requested on CPU should warn (fall back to fp32) and still run."""
    model = _build_model()
    records = [PromptCompletion(prompt="alpha", completion=" beta")]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = train_thinker(
            model,
            records,
            _FakeTokenizer(),
            TrainConfig(
                max_length=16,
                batch_size=1,
                lr=1e-2,
                num_epochs=1,
                thinker_steps=1,
                log_every=0,
                device="cpu",
                precision="fp16",
            ),
            output_dir=str(tmp_path / "hats"),
        )

    assert result.steps >= 1
    assert any(
        issubclass(w.category, UserWarning) and "fp16" in str(w.message)
        for w in caught
    )
