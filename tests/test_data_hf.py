"""HF datasets adapter -> PromptCompletion records (issue #13).

Schema: each row exposes non-empty string `prompt_field` and `completion_field`
columns; extra columns are ignored. The adapter is lazy and fails fast on the
first malformed row with a message naming the dataset/split.

Tests never hit the network: `datasets.load_dataset` is monkeypatched to return
real in-memory `Dataset` / `IterableDataset` objects (PRD Testing Decisions).
"""

from __future__ import annotations

import pytest

datasets = pytest.importorskip("datasets")

from hugging_hat.data import (  # noqa: E402
    InvalidDatasetError,
    PromptCompletion,
    load_hf_dataset,
)


def _patch_load_dataset(monkeypatch, ds):
    """Make `datasets.load_dataset` return `ds`; capture its call args."""
    calls: list[dict] = []

    def fake_load_dataset(path, name=None, split=None, streaming=False, **kw):
        calls.append(
            dict(path=path, name=name, split=split, streaming=streaming, kw=kw)
        )
        return ds

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    return calls


def test_yields_prompt_completion_records_and_forwards_load_args(monkeypatch):
    ds = datasets.Dataset.from_dict(
        {"prompt": ["2+2=", "capital of France?"], "completion": ["4", "Paris"]}
    )
    calls = _patch_load_dataset(monkeypatch, ds)

    pairs = list(
        load_hf_dataset("dummy/ds", split="train", config="cfg-A", streaming=False)
    )

    assert pairs == [
        PromptCompletion(prompt="2+2=", completion="4"),
        PromptCompletion(prompt="capital of France?", completion="Paris"),
    ]
    assert calls == [
        dict(path="dummy/ds", name="cfg-A", split="train", streaming=False, kw={})
    ]


def test_custom_field_names_resolve_correct_columns(monkeypatch):
    ds = datasets.Dataset.from_dict(
        {"question": ["q1", "q2"], "answer": ["a1", "a2"], "noise": [0, 0]}
    )
    _patch_load_dataset(monkeypatch, ds)

    pairs = list(
        load_hf_dataset(
            "dummy/ds",
            split="train",
            prompt_field="question",
            completion_field="answer",
        )
    )

    assert pairs == [
        PromptCompletion(prompt="q1", completion="a1"),
        PromptCompletion(prompt="q2", completion="a2"),
    ]


def test_missing_column_fails_fast_with_dataset_split_and_column(monkeypatch):
    ds = datasets.Dataset.from_dict(
        {"prompt": ["p1"], "answer": ["a1"]}  # no `completion`
    )
    _patch_load_dataset(monkeypatch, ds)

    with pytest.raises(InvalidDatasetError) as exc:
        list(load_hf_dataset("acme/qa", split="validation"))

    msg = str(exc.value)
    assert "acme/qa" in msg, msg
    assert "validation" in msg, msg
    assert "completion" in msg, msg
    assert "missing" in msg.lower(), msg


def test_empty_string_value_is_rejected(monkeypatch):
    ds = datasets.Dataset.from_dict(
        {"prompt": ["ok", "still ok"], "completion": ["fine", ""]}
    )
    _patch_load_dataset(monkeypatch, ds)

    gen = load_hf_dataset("acme/qa", split="train")
    # First record is valid — adapter is lazy.
    assert next(gen) == PromptCompletion(prompt="ok", completion="fine")
    with pytest.raises(InvalidDatasetError) as exc:
        next(gen)

    msg = str(exc.value)
    assert "acme/qa" in msg, msg
    assert "completion" in msg and "non-empty" in msg, msg


def test_non_string_value_is_rejected_with_type_info(monkeypatch):
    ds = datasets.Dataset.from_dict({"prompt": ["bad"], "completion": [42]})
    _patch_load_dataset(monkeypatch, ds)

    with pytest.raises(InvalidDatasetError) as exc:
        list(load_hf_dataset("acme/qa", split="train"))

    msg = str(exc.value)
    assert "acme/qa" in msg, msg
    assert "completion" in msg, msg
    assert "string" in msg and "int" in msg, msg


def test_streaming_path_iterates_iterable_dataset(monkeypatch):
    def gen():
        yield {"prompt": "s1", "completion": "c1"}
        yield {"prompt": "s2", "completion": "c2"}

    ds = datasets.IterableDataset.from_generator(gen)
    calls = _patch_load_dataset(monkeypatch, ds)

    pairs = list(load_hf_dataset("dummy/ds", split="train", streaming=True))

    assert pairs == [
        PromptCompletion(prompt="s1", completion="c1"),
        PromptCompletion(prompt="s2", completion="c2"),
    ]
    assert calls[0]["streaming"] is True


def test_public_api_is_importable_from_package_root():
    import hugging_hat

    assert hugging_hat.load_hf_dataset is load_hf_dataset
