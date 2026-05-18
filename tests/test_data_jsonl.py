"""JSONL prompt/completion dataset loader + schema validation (issue #5).

Schema: each line is a JSON object with non-empty string `prompt` and
`completion`; extra keys are ignored. The loader is lazy and fails fast on
the first malformed line with a `file:line - reason` message.
"""

from __future__ import annotations

import pytest

from hugging_hat.data import (
    InvalidDatasetError,
    PromptCompletion,
    load_jsonl,
)


def test_public_api_is_importable_from_package_root():
    import hugging_hat

    assert hugging_hat.load_jsonl is load_jsonl
    assert hugging_hat.PromptCompletion is PromptCompletion
    assert hugging_hat.InvalidDatasetError is InvalidDatasetError


def _write(tmp_path, *lines: str):
    p = tmp_path / "data.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_valid_jsonl_yields_prompt_completion_pairs(tmp_path):
    path = _write(
        tmp_path,
        '{"prompt": "2+2=", "completion": "4"}',
        '{"prompt": "capital of France?", "completion": "Paris"}',
    )

    pairs = list(load_jsonl(str(path)))

    assert pairs == [
        PromptCompletion(prompt="2+2=", completion="4"),
        PromptCompletion(prompt="capital of France?", completion="Paris"),
    ]


def test_missing_required_key_fails_fast_with_location(tmp_path):
    path = _write(
        tmp_path,
        '{"prompt": "ok", "completion": "fine"}',
        '{"prompt": "no completion here"}',
    )

    with pytest.raises(InvalidDatasetError) as exc:
        list(load_jsonl(str(path)))

    msg = str(exc.value)
    assert "data.jsonl:2" in msg, msg
    assert "completion" in msg, msg


def test_non_string_value_fails_fast_with_helpful_message(tmp_path):
    path = _write(
        tmp_path,
        '{"prompt": "ok", "completion": "fine"}',
        '{"prompt": "bad type", "completion": 42}',
    )

    with pytest.raises(InvalidDatasetError) as exc:
        list(load_jsonl(str(path)))

    msg = str(exc.value)
    assert "data.jsonl:2" in msg, msg
    assert "completion" in msg, msg
    assert "string" in msg and "int" in msg, msg


def test_empty_string_value_is_rejected(tmp_path):
    path = _write(
        tmp_path,
        '{"prompt": "has prompt", "completion": ""}',
    )

    with pytest.raises(InvalidDatasetError) as exc:
        list(load_jsonl(str(path)))

    msg = str(exc.value)
    assert "data.jsonl:1" in msg, msg
    assert "completion" in msg and "non-empty" in msg, msg


def test_invalid_json_fails_fast_with_line_number(tmp_path):
    path = _write(
        tmp_path,
        '{"prompt": "ok", "completion": "fine"}',
        "{not valid json",
    )

    with pytest.raises(InvalidDatasetError) as exc:
        list(load_jsonl(str(path)))

    msg = str(exc.value)
    assert "data.jsonl:2" in msg, msg
    assert "JSON" in msg, msg


def test_extra_keys_are_ignored(tmp_path):
    path = _write(
        tmp_path,
        '{"prompt": "p", "completion": "c", "id": 7, "source": "synthetic"}',
    )

    pairs = list(load_jsonl(str(path)))

    assert pairs == [PromptCompletion(prompt="p", completion="c")]


def test_blank_lines_are_skipped(tmp_path):
    path = _write(
        tmp_path,
        '{"prompt": "a", "completion": "1"}',
        "",
        "   ",
        '{"prompt": "b", "completion": "2"}',
    )

    pairs = list(load_jsonl(str(path)))

    assert pairs == [
        PromptCompletion(prompt="a", completion="1"),
        PromptCompletion(prompt="b", completion="2"),
    ]


def test_loader_is_lazy(tmp_path):
    # Second line is malformed; pulling only the first item must NOT raise,
    # proving the loader streams rather than validating the whole file upfront.
    path = _write(
        tmp_path,
        '{"prompt": "first", "completion": "ok"}',
        "{ broken",
    )

    gen = load_jsonl(str(path))
    first = next(gen)

    assert first == PromptCompletion(prompt="first", completion="ok")
    with pytest.raises(InvalidDatasetError):
        next(gen)


def test_non_object_line_fails_fast(tmp_path):
    # Valid JSON, but not an object: schema validation must still reject it
    # cleanly rather than leaking a TypeError.
    path = _write(
        tmp_path,
        '["prompt", "completion"]',
    )

    with pytest.raises(InvalidDatasetError) as exc:
        list(load_jsonl(str(path)))

    msg = str(exc.value)
    assert "data.jsonl:1" in msg, msg
    assert "object" in msg, msg
