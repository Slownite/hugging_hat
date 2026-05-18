from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open

from hugging_hat.config import (
    HatConfig,
    LatentRouterHatConfig,
    ThinkerHatConfig,
    hat_config_from_dict,
)
from hugging_hat.persistence import (
    IncompatibleHatCheckpointError,
    InvalidHatCheckpointError,
)
from hugging_hat.model import HatEnabledModel
from hugging_hat.testing.dummy_hf import build_dummy_causallm


def _thinker_only_config() -> HatConfig:
    # Thinker active (default_steps>0) so hat weights actually affect outputs;
    # router/critic disabled to keep the round-trip deterministic.
    return HatConfig(
        layers_path="model.layers",
        thinker=ThinkerHatConfig(enabled=True, default_steps=2),
        router=LatentRouterHatConfig(enabled=False),
    )


def _wrap(*, base_seed: int, hat_seed: int) -> HatEnabledModel:
    torch.manual_seed(base_seed)
    base = build_dummy_causallm(hidden_size=32, num_hidden_layers=8)
    torch.manual_seed(hat_seed)
    return HatEnabledModel(base, config=_thinker_only_config())


def test_save_load_round_trip_is_bit_identical(tmp_path):
    ids = torch.randint(0, 128, (2, 16))
    mask = torch.ones_like(ids)

    trained = _wrap(base_seed=0, hat_seed=1)
    trained.eval()
    with torch.no_grad():
        reference = trained(input_ids=ids, attention_mask=mask)["logits"]

    ckpt = tmp_path / "hat_ckpt"
    trained.save_hats(str(ckpt))

    # Fresh instance: identical base weights (same base_seed), but a
    # differently-initialised thinker hat so loading must actually change it.
    fresh = _wrap(base_seed=0, hat_seed=2)
    fresh.eval()
    with torch.no_grad():
        before = fresh(input_ids=ids, attention_mask=mask)["logits"]
    assert not torch.equal(before, reference), "sanity: fresh hat must differ pre-load"

    fresh.load_hats(str(ckpt))
    with torch.no_grad():
        after = fresh(input_ids=ids, attention_mask=mask)["logits"]

    assert torch.equal(after, reference)


def test_only_hat_tensors_are_saved(tmp_path):
    model = _wrap(base_seed=0, hat_seed=1)
    ckpt = tmp_path / "hat_ckpt"
    model.save_hats(str(ckpt))

    with safe_open(str(ckpt / "hats.safetensors"), framework="pt") as f:
        keys = list(f.keys())

    assert keys, "expected at least the thinker hat tensors"
    assert all(k.startswith("thinker.") for k in keys), keys
    # No base-model parameters leaked in.
    base_params = {n for n, _ in model.base_model.named_parameters()}
    assert not (set(keys) & base_params)
    assert not any(k.startswith(("base_model.", "embed.", "lm_head.")) for k in keys)


def test_save_writes_config_and_metadata(tmp_path):
    model = _wrap(base_seed=0, hat_seed=1)
    ckpt = tmp_path / "hat_ckpt"
    model.save_hats(str(ckpt))

    cfg = json.loads((ckpt / "hat_config.json").read_text())
    assert hat_config_from_dict(cfg) == model.config

    meta = json.loads((ckpt / "hat_metadata.json").read_text())
    for key in (
        "format_version",
        "library_version",
        "created_at",
        "base_model",
        "hats_present",
        "layer_attachment",
        "step_set",
        "tensor_format",
        "tensors",
    ):
        assert key in meta, f"missing metadata key: {key}"

    assert meta["format_version"] == 1
    assert meta["tensor_format"] == "safetensors"
    assert meta["hats_present"] == ["thinker"]
    assert meta["base_model"]["hidden_size"] == 32
    assert meta["base_model"]["num_hidden_layers"] == 8
    assert meta["layer_attachment"]["thinker"]["resolved_index"] == model.thinker_layer_idx
    assert meta["step_set"] == list(model.config.thinker.step_set)
    assert meta["tensors"]["thinker"]["num_params"] > 0


def test_missing_checkpoint_raises_invalid(tmp_path):
    model = _wrap(base_seed=0, hat_seed=1)
    with pytest.raises(InvalidHatCheckpointError, match="missing hats.safetensors"):
        model.load_hats(str(tmp_path / "does_not_exist"))


def test_hidden_size_mismatch_raises_incompatible(tmp_path):
    trained = _wrap(base_seed=0, hat_seed=1)  # hidden_size=32
    ckpt = tmp_path / "hat_ckpt"
    trained.save_hats(str(ckpt))

    torch.manual_seed(0)
    big_base = build_dummy_causallm(hidden_size=64, num_hidden_layers=8)
    target = HatEnabledModel(big_base, config=_thinker_only_config())

    with pytest.raises(IncompatibleHatCheckpointError, match="hidden_size mismatch"):
        target.load_hats(str(ckpt))


def test_component_present_but_disabled_raises_incompatible(tmp_path):
    trained = _wrap(base_seed=0, hat_seed=1)  # thinker enabled
    ckpt = tmp_path / "hat_ckpt"
    trained.save_hats(str(ckpt))

    torch.manual_seed(0)
    base = build_dummy_causallm(hidden_size=32, num_hidden_layers=8)
    target = HatEnabledModel(
        base,
        config=HatConfig(
            layers_path="model.layers",
            thinker=ThinkerHatConfig(enabled=False),
            router=LatentRouterHatConfig(enabled=False),
        ),
    )
    with pytest.raises(IncompatibleHatCheckpointError, match="thinker.*enabled is False"):
        target.load_hats(str(ckpt))


def test_thinker_shape_mismatch_raises_incompatible(tmp_path):
    trained = _wrap(base_seed=0, hat_seed=1)  # default hidden_multiplier=4
    ckpt = tmp_path / "hat_ckpt"
    trained.save_hats(str(ckpt))

    torch.manual_seed(0)
    base = build_dummy_causallm(hidden_size=32, num_hidden_layers=8)
    target = HatEnabledModel(
        base,
        config=HatConfig(
            layers_path="model.layers",
            thinker=ThinkerHatConfig(enabled=True, default_steps=2, hidden_multiplier=8),
            router=LatentRouterHatConfig(enabled=False),
        ),
    )
    with pytest.raises(IncompatibleHatCheckpointError, match="[Ss]hape mismatch"):
        target.load_hats(str(ckpt))


def test_library_version_mismatch_warns_but_loads(tmp_path):
    trained = _wrap(base_seed=0, hat_seed=1)
    trained.eval()
    ids = torch.randint(0, 128, (2, 16))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        reference = trained(input_ids=ids, attention_mask=mask)["logits"]

    ckpt = tmp_path / "hat_ckpt"
    trained.save_hats(str(ckpt))

    meta_path = ckpt / "hat_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["library_version"] = "0.0.0-ancient"
    meta_path.write_text(json.dumps(meta))

    fresh = _wrap(base_seed=0, hat_seed=2)
    fresh.eval()
    with pytest.warns(UserWarning, match="Behavior may differ"):
        fresh.load_hats(str(ckpt))

    with torch.no_grad():
        after = fresh(input_ids=ids, attention_mask=mask)["logits"]
    assert torch.equal(after, reference)
