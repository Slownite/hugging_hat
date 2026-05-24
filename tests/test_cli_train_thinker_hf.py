"""Tests for `hh train-thinker --dataset ...` (issue #14).

Mirrors the JSONL CLI tests in test_cli_train_thinker.py, but exercises the
Hugging Face datasets adapter source. Never hits the network: monkeypatches
``datasets.load_dataset`` to return an in-memory ``datasets.Dataset``.
"""

from __future__ import annotations

import pytest
import torch
from click.testing import CliRunner

datasets = pytest.importorskip("datasets")

from hugging_hat.cli import cli  # noqa: E402
from hugging_hat.config import HatConfig, LatentRouterHatConfig, ThinkerHatConfig  # noqa: E402
from hugging_hat.model import HatEnabledModel  # noqa: E402
from hugging_hat.testing.dummy_hf import build_dummy_causallm  # noqa: E402


# ----------------------------------------------------------------------- fixtures

class _FakeTokenizer:
    """Whitespace tokenizer that satisfies preprocess_record's contract."""

    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = vocab_size
        self.bos_id = 1
        self.eos_id = 2
        self.pad_token_id: int | None = 0
        self.eos_token_id: int = self.eos_id

    def _tokens(self, text: str) -> list[int]:
        return [3 + (abs(hash(w)) % (self.vocab_size - 3)) for w in text.split()]

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


def _hat_config() -> HatConfig:
    return HatConfig(
        layers_path="model.layers",
        thinker=ThinkerHatConfig(enabled=True, default_steps=0),
        router=LatentRouterHatConfig(enabled=False),
    )


def _patch_loaders(monkeypatch) -> dict:
    handles: dict = {}

    def _fake_model_loader(model_name_or_path, *, config=None, **kwargs):
        torch.manual_seed(0)
        base = build_dummy_causallm(hidden_size=32, num_hidden_layers=4)
        torch.manual_seed(1)
        model = HatEnabledModel(base, config=config or _hat_config())
        handles["model"] = model
        return model

    def _fake_tokenizer_loader(model_name_or_path, *args, **kwargs):
        return _FakeTokenizer()

    monkeypatch.setattr(
        HatEnabledModel, "from_pretrained", classmethod(
            lambda cls, *a, **kw: _fake_model_loader(*a, **kw)
        )
    )
    import transformers
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", _fake_tokenizer_loader
    )
    return handles


def _patch_hf_load_dataset(monkeypatch, ds) -> list[dict]:
    calls: list[dict] = []

    def fake_load_dataset(path, name=None, split=None, streaming=False, **kw):
        calls.append(
            dict(path=path, name=name, split=split, streaming=streaming, kw=kw)
        )
        return ds

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    return calls


# ------------------------------------------------------------------------- tests

def test_cli_train_thinker_with_dataset_writes_loadable_hat_checkpoint(
    tmp_path, monkeypatch
):
    _patch_loaders(monkeypatch)
    ds = datasets.Dataset.from_dict(
        {
            "prompt": ["alpha beta", "hello"],
            "completion": [" gamma delta", " world"],
        }
    )
    _patch_hf_load_dataset(monkeypatch, ds)

    out_dir = tmp_path / "hats"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "train-thinker",
            "dummy-model",
            "--dataset", "dummy/ds",
            "--split", "train",
            "--output-dir", str(out_dir),
            "--thinker-steps", "2",
            "--max-length", "32",
            "--batch-size", "2",
            "--lr", "1e-2",
            "--epochs", "1",
            "--log-every", "0",
            "--device", "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "hats.safetensors").is_file()
    assert (out_dir / "hat_metadata.json").is_file()

    torch.manual_seed(0)
    fresh_base = build_dummy_causallm(hidden_size=32, num_hidden_layers=4)
    fresh = HatEnabledModel(fresh_base, config=HatConfig())
    fresh.load_hats(str(out_dir))


def test_cli_train_thinker_forwards_hf_flags_to_load_dataset(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    # Use custom column names + a dataset config name + a non-train split to
    # verify every HF flag round-trips into datasets.load_dataset.
    ds = datasets.Dataset.from_dict(
        {"question": ["q1", "q2"], "answer": ["a1", "a2"], "noise": [0, 0]}
    )
    calls = _patch_hf_load_dataset(monkeypatch, ds)

    out_dir = tmp_path / "hats"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "train-thinker",
            "dummy-model",
            "--dataset", "acme/qa",
            "--split", "validation",
            "--dataset-config", "cfg-A",
            "--prompt-field", "question",
            "--completion-field", "answer",
            "--output-dir", str(out_dir),
            "--thinker-steps", "2",
            "--max-length", "16",
            "--batch-size", "2",
            "--lr", "1e-2",
            "--epochs", "1",
            "--log-every", "0",
            "--device", "cpu",
        ],
    )
    assert result.exit_code == 0, result.output

    assert calls, "datasets.load_dataset should have been called"
    call = calls[0]
    assert call["path"] == "acme/qa"
    assert call["name"] == "cfg-A"
    assert call["split"] == "validation"
    assert call["streaming"] is False


def test_cli_train_thinker_rejects_both_data_and_dataset(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)

    data = tmp_path / "train.jsonl"
    data.write_text('{"prompt": "p", "completion": "c"}\n')
    out_dir = tmp_path / "hats"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "train-thinker",
            "dummy-model",
            "--data", str(data),
            "--dataset", "acme/qa",
            "--split", "train",
            "--output-dir", str(out_dir),
        ],
    )

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    out = result.output.lower()
    assert "--data" in out and "--dataset" in out
    assert (
        "mutually exclusive" in out
        or "cannot" in out
        or "exactly one" in out
    ), result.output


def test_cli_train_thinker_requires_data_or_dataset(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    out_dir = tmp_path / "hats"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "train-thinker",
            "dummy-model",
            "--output-dir", str(out_dir),
        ],
    )

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    out = result.output.lower()
    assert "--data" in out and "--dataset" in out
    assert "exactly one" in out or "required" in out, result.output
