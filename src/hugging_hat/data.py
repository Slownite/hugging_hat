from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


class InvalidDatasetError(RuntimeError):
    """A dataset line is not a structurally valid prompt/completion record."""


@dataclass(frozen=True, slots=True)
class PromptCompletion:
    prompt: str
    completion: str


_REQUIRED_KEYS = ("prompt", "completion")


def load_jsonl(path: str) -> Iterator[PromptCompletion]:
    name = Path(path).name
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise InvalidDatasetError(
                    f"{name}:{lineno} - invalid JSON: {e.msg}"
                ) from e
            if not isinstance(obj, dict):
                raise InvalidDatasetError(
                    f"{name}:{lineno} - line must be a JSON object, "
                    f"got {type(obj).__name__}"
                )
            for key in _REQUIRED_KEYS:
                if key not in obj:
                    raise InvalidDatasetError(
                        f"{name}:{lineno} - missing required key {key!r}"
                    )
                value = obj[key]
                if not isinstance(value, str):
                    raise InvalidDatasetError(
                        f"{name}:{lineno} - {key!r} must be a string, "
                        f"got {type(value).__name__}"
                    )
                if not value:
                    raise InvalidDatasetError(
                        f"{name}:{lineno} - {key!r} must be a non-empty string"
                    )
            yield PromptCompletion(prompt=obj["prompt"], completion=obj["completion"])
