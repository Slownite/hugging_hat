"""Deterministic dummy-model round-trip smoke test (issue #4).

Distinct from test_hat_persistence.py (which unit-tests save/load mechanics
and error paths): this is the end-to-end smoke test that drives the *full*
multi-step generate path with a fixed seed, persists hats to a temp dir,
rebuilds a fresh instance, loads, and asserts the generated token stream is
reproduced exactly.
"""

from __future__ import annotations

import torch

from hugging_hat.config import (
    HatConfig,
    LatentRouterHatConfig,
    ThinkerHatConfig,
)
from hugging_hat.model import HatEnabledModel
from hugging_hat.testing.dummy_hf import build_dummy_causallm


def _config() -> HatConfig:
    # Thinker active so hat weights actually influence the output stream;
    # router/critic disabled so the round-trip stays deterministic.
    return HatConfig(
        layers_path="model.layers",
        thinker=ThinkerHatConfig(enabled=True, default_steps=3),
        router=LatentRouterHatConfig(enabled=False),
    )


def _build(*, base_seed: int, hat_seed: int) -> HatEnabledModel:
    torch.manual_seed(base_seed)
    base = build_dummy_causallm(hidden_size=32, num_hidden_layers=8)
    torch.manual_seed(hat_seed)
    model = HatEnabledModel(base, config=_config())
    model.eval()
    return model


def test_dummy_model_generate_round_trip_is_reproduced(tmp_path):
    ids = torch.randint(0, 128, (2, 8))
    mask = torch.ones_like(ids)

    trained = _build(base_seed=0, hat_seed=1)
    with torch.no_grad():
        reference = trained.generate(
            input_ids=ids, attention_mask=mask, max_new_tokens=5
        )

    ckpt = tmp_path / "hat_ckpt"
    trained.save_hats(str(ckpt))

    # Fresh instance: identical base weights (same base_seed) but a
    # differently-seeded thinker hat, so loading must actually change behavior.
    fresh = _build(base_seed=0, hat_seed=2)
    with torch.no_grad():
        before = fresh.generate(
            input_ids=ids, attention_mask=mask, max_new_tokens=5
        )
    assert not torch.equal(before, reference), (
        "sanity: a freshly-initialised hat must change the generated stream"
    )

    fresh.load_hats(str(ckpt))
    with torch.no_grad():
        after = fresh.generate(
            input_ids=ids, attention_mask=mask, max_new_tokens=5
        )

    assert torch.equal(after, reference), (
        "loaded hats must reproduce the original generated token stream exactly"
    )
