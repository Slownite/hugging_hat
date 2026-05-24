"""Determinism controls for v0 Thinker training (issue #10).

`set_seed(None)` is intentionally a no-op so callers can pass a config field
straight through without branching. `set_seed(int)` seeds every RNG we may
touch and flips on torch's deterministic-algorithm flags so two runs with the
same seed and inputs produce identical loss trajectories.
"""

from __future__ import annotations

import os
import random

import torch


def set_seed(seed: int | None) -> None:
    if seed is None:
        return

    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # CuBLAS needs this env var set before its workspace is allocated for
    # `use_deterministic_algorithms(True)` to succeed on CUDA. Setting it here
    # is best-effort: if cuBLAS has already been initialised it will be ignored,
    # but for fresh runs (the common case) it avoids a RuntimeError.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
