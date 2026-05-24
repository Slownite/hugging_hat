"""Tests for `set_seed` (issue #10).

Determinism controls for v0 Thinker training. Seeds python ``random``,
``numpy`` (when present), and torch RNGs; sets torch deterministic flags.
``set_seed(None)`` is a no-op that does not touch global state.
"""

from __future__ import annotations

import random

import torch

from hugging_hat.train import set_seed


def test_set_seed_reproduces_torch_rand():
    set_seed(0)
    a = torch.rand(5)
    set_seed(0)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_set_seed_reproduces_python_random():
    set_seed(0)
    a = [random.random() for _ in range(5)]
    set_seed(0)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_seeds_numpy_when_present():
    np = pytest_importorskip_numpy()
    set_seed(0)
    a = np.random.rand(5)
    set_seed(0)
    b = np.random.rand(5)
    assert (a == b).all()


def test_set_seed_none_does_not_change_torch_rng_state():
    torch.manual_seed(123)
    snap = torch.get_rng_state()
    set_seed(None)
    assert torch.equal(snap, torch.get_rng_state()), (
        "set_seed(None) must not touch the torch RNG state"
    )


def test_set_seed_none_does_not_enable_deterministic_algorithms():
    # Reset deterministic flag to a known value before the assertion.
    torch.use_deterministic_algorithms(False)
    set_seed(None)
    assert torch.are_deterministic_algorithms_enabled() is False


def test_set_seed_int_enables_deterministic_algorithms():
    try:
        set_seed(0)
        assert torch.are_deterministic_algorithms_enabled() is True
    finally:
        # Don't pollute other tests with the global flag.
        torch.use_deterministic_algorithms(False)


def pytest_importorskip_numpy():
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a torch dep
        import pytest

        pytest.skip("numpy is not installed")
    return np
