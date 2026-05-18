from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


class LayerStackNotFoundError(RuntimeError):
    pass


def _getattr_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def _is_module_stack(obj: Any) -> bool:
    # torch.nn.ModuleList is the canonical decoder stack but is NOT a
    # collections.abc.Sequence, so duck-type: indexable, sized, holds Modules.
    try:
        n = len(obj)
    except TypeError:
        return False
    if n == 0:
        return False
    try:
        first = obj[0]
    except (TypeError, KeyError, IndexError):
        return False
    return isinstance(first, nn.Module)


def resolve_decoder_layers(model: nn.Module, *, layers_path: str | None = None) -> tuple[Sequence[nn.Module], str]:
    if layers_path is not None:
        layers = _getattr_path(model, layers_path)
        if not _is_module_stack(layers):
            raise TypeError(
                f"layers_path={layers_path!r} did not resolve to a non-empty stack of nn.Modules"
            )
        return layers, layers_path

    candidates = [
        "model.layers",  # LLaMA-style
        "model.model.layers",  # some wrappers
        "layers",  # simple
        "transformer.h",  # GPT-2 style
        "gpt_neox.layers",  # GPT-NeoX style
        "model.decoder.layers",  # some encoder/decoder-ish wrappers
    ]

    for path in candidates:
        try:
            layers = _getattr_path(model, path)
        except AttributeError:
            continue
        if _is_module_stack(layers):
            return layers, path

    raise LayerStackNotFoundError(
        "Could not resolve decoder layer stack. "
        "Pass HatConfig(layers_path=...) to specify where the block list lives."
    )


def is_prefill_forward(*, past_key_values: Any | None, input_ids: torch.Tensor | None) -> bool:
    if past_key_values is None:
        return True
    if input_ids is None:
        return False
    return input_ids.shape[1] > 1

