"""Tests for the `hh train-thinker` CLI command (issue #12)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from click.testing import CliRunner

from hugging_hat.cli import cli
from hugging_hat.config import HatConfig, LatentRouterHatConfig, ThinkerHatConfig
from hugging_hat.model import HatEnabledModel
from hugging_hat.testing.dummy_hf import build_dummy_causallm


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


def _build_hat_model() -> HatEnabledModel:
    torch.manual_seed(0)
    base = build_dummy_causallm(hidden_size=32, num_hidden_layers=4)
    torch.manual_seed(1)
    return HatEnabledModel(base, config=_hat_config())


def _write_jsonl(path: Path) -> None:
    rows = [
        {"prompt": "alpha beta", "completion": " gamma delta"},
        {"prompt": "hello", "completion": " world"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _patch_loaders(monkeypatch) -> dict:
    """Patch model + tokenizer loaders to return in-memory dummies. Returns the
    handles so tests can inspect them post-invocation.
    """
    handles: dict = {}

    def _fake_model_loader(model_name_or_path, *, config=None, **kwargs):
        torch.manual_seed(0)
        base = build_dummy_causallm(hidden_size=32, num_hidden_layers=4)
        torch.manual_seed(1)
        model = HatEnabledModel(base, config=config or _hat_config())
        handles["model"] = model
        handles["model_kwargs"] = kwargs
        handles["model_path"] = model_name_or_path
        handles["hat_config"] = config
        return model

    def _fake_tokenizer_loader(model_name_or_path, *args, **kwargs):
        tok = _FakeTokenizer()
        handles["tokenizer"] = tok
        handles["tokenizer_kwargs"] = kwargs
        return tok

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


# ------------------------------------------------------------------------- tests

def test_cli_train_thinker_writes_loadable_hat_checkpoint(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)

    data = tmp_path / "train.jsonl"
    _write_jsonl(data)
    out_dir = tmp_path / "hats"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "train-thinker",
            "dummy-model",
            "--data", str(data),
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

    # The checkpoint must be loadable onto a fresh HatEnabledModel built with
    # the same config the CLI used (default HatConfig in this invocation).
    torch.manual_seed(0)
    fresh_base = build_dummy_causallm(hidden_size=32, num_hidden_layers=4)
    fresh = HatEnabledModel(fresh_base, config=HatConfig())
    fresh.load_hats(str(out_dir))


def test_cli_train_thinker_surfaces_missing_optional_dep_as_click_exception(
    tmp_path, monkeypatch
):
    data = tmp_path / "train.jsonl"
    _write_jsonl(data)
    out_dir = tmp_path / "hats"

    def _raise_missing(cls, *a, **kw):
        raise ModuleNotFoundError(
            "Hugging Face integration requires optional dependency: "
            "`pip install hugging-hat[hf]`"
        )

    monkeypatch.setattr(
        HatEnabledModel, "from_pretrained", classmethod(_raise_missing)
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "train-thinker",
            "dummy-model",
            "--data", str(data),
            "--output-dir", str(out_dir),
        ],
    )

    assert result.exit_code != 0
    # ClickException prints "Error: <msg>" to stderr/output, not a traceback.
    assert "Traceback" not in result.output
    assert "Error:" in result.output
    assert "hugging-hat[hf]" in result.output


def test_cli_train_thinker_resume_from_restores_hat_weights(tmp_path, monkeypatch):
    handles = _patch_loaders(monkeypatch)

    data = tmp_path / "train.jsonl"
    _write_jsonl(data)
    first_out = tmp_path / "hats_a"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "train-thinker",
            "dummy-model",
            "--data", str(data),
            "--output-dir", str(first_out),
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

    trained_hat_state = {
        n: p.detach().clone()
        for n, p in handles["model"].named_parameters()
        if n.startswith(("thinker_hat.", "router_hat.", "critic_hat."))
    }
    assert trained_hat_state, "fixture must enable at least one hat"

    # Second run: resume from the saved checkpoint and immediately save without
    # training. The freshly-built fake model has a different hat init; if
    # --resume-from is honored, load_hats must overwrite it with the trained
    # state before save_hats writes the second checkpoint.
    second_out = tmp_path / "hats_b"
    result2 = runner.invoke(
        cli,
        [
            "train-thinker",
            "dummy-model",
            "--data", str(data),
            "--output-dir", str(second_out),
            "--thinker-steps", "2",
            "--max-length", "16",
            "--batch-size", "2",
            "--epochs", "0",  # no training; just resume + save
            "--log-every", "0",
            "--device", "cpu",
            "--resume-from", str(first_out),
        ],
    )
    assert result2.exit_code == 0, result2.output

    resumed_model = handles["model"]
    for name, value in trained_hat_state.items():
        loaded = dict(resumed_model.named_parameters())[name].detach()
        assert torch.equal(loaded, value), (
            f"hat param {name} should have been restored by --resume-from"
        )


def test_cli_train_thinker_honors_config_yaml(tmp_path, monkeypatch):
    handles = _patch_loaders(monkeypatch)

    data = tmp_path / "train.jsonl"
    _write_jsonl(data)
    out_dir = tmp_path / "hats"

    config_yaml = tmp_path / "hats.yaml"
    config_yaml.write_text(
        "layers_path: model.layers\n"
        "thinker:\n"
        "  enabled: true\n"
        "  attach_layer: 1\n"
        "  step_set: [0, 1, 2]\n"
        "router:\n"
        "  enabled: false\n"
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "train-thinker",
            "dummy-model",
            "--data", str(data),
            "--output-dir", str(out_dir),
            "--config-yaml", str(config_yaml),
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

    # The YAML's thinker.attach_layer=1 must show up in pre-flight, and
    # step_set must reflect the YAML's [0, 1, 2].
    assert "thinker_layer_idx: 1" in result.output
    assert "step_set: [0, 1, 2]" in result.output

    # The loader received the parsed HatConfig (not the default).
    received = handles["hat_config"]
    assert received is not None
    assert received.thinker.attach_layer == 1
    assert received.thinker.step_set == (0, 1, 2)
    assert received.router.enabled is False


def test_cli_train_thinker_prints_preflight_summary(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)

    data = tmp_path / "train.jsonl"
    _write_jsonl(data)
    out_dir = tmp_path / "hats"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "train-thinker",
            "dummy-model",
            "--data", str(data),
            "--output-dir", str(out_dir),
            "--thinker-steps", "3",
            "--max-length", "16",
            "--batch-size", "2",
            "--lr", "1e-2",
            "--epochs", "1",
            "--log-every", "0",
            "--device", "cpu",
            "--precision", "fp32",
        ],
    )
    assert result.exit_code == 0, result.output

    out = result.output
    # Required fields from training-api-v0.md pre-flight (PRD story 14).
    assert "layers_path" in out
    assert "model.layers" in out  # the configured layers_path value
    assert "thinker_layer_idx" in out
    assert "router_layer_idx" in out
    assert "critic_layer_idx" in out
    assert "thinker_steps" in out and "3" in out
    assert "step_set" in out
    assert "device" in out and "cpu" in out
    assert "precision" in out and "fp32" in out
    assert "batch_size" in out and "2" in out
    assert "max_length" in out and "16" in out
