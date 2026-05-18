from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import load_file as _st_load
from safetensors.torch import save_file as _st_save

if TYPE_CHECKING:
    from .model import HatEnabledModel


class InvalidHatCheckpointError(RuntimeError):
    """The artifact is not a structurally valid hat checkpoint."""


class IncompatibleHatCheckpointError(RuntimeError):
    """The checkpoint cannot be safely loaded onto the target model/config."""


FORMAT_VERSION = 1

_WEIGHTS_FILE = "hats.safetensors"
_CONFIG_FILE = "hat_config.json"
_METADATA_FILE = "hat_metadata.json"

_COMPONENTS = ("thinker", "router", "critic")


def _library_version() -> str:
    try:
        return version("hugging-hat")
    except PackageNotFoundError:  # pragma: no cover - dev/source checkouts
        return "0.0.0+unknown"


def _hat_modules(model: "HatEnabledModel") -> dict[str, torch.nn.Module]:
    out: dict[str, torch.nn.Module] = {}
    for name in _COMPONENTS:
        hat = getattr(model, f"{name}_hat", None)
        if hat is not None:
            out[name] = hat
    return out


def _base_model_info(model: "HatEnabledModel") -> dict[str, Any]:
    base_cfg = model.base_model.config
    name_or_path = (
        getattr(base_cfg, "_name_or_path", None)
        or getattr(model.base_model, "name_or_path", None)
        or ""
    )
    model_type = getattr(base_cfg, "model_type", None) or type(model.base_model).__name__
    return {
        "name_or_path": name_or_path,
        "model_type": model_type,
        "hidden_size": int(model.hidden_size),
        "num_hidden_layers": int(model.num_layers),
    }


def _layer_attachment(model: "HatEnabledModel", present: list[str]) -> dict[str, Any]:
    selectors = {
        "thinker": (model.config.thinker.attach_layer, model.thinker_layer_idx),
        "router": (model.config.router.attach_layer, model.router_layer_idx),
        "critic": (model.config.critic.attach_layer, model.critic_layer_idx),
    }
    return {
        name: {"selector": selectors[name][0], "resolved_index": int(selectors[name][1])}
        for name in present
    }


def save_hats(model: "HatEnabledModel", path: str) -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)

    modules = _hat_modules(model)
    present = list(modules)

    tensors: dict[str, torch.Tensor] = {}
    tensor_meta: dict[str, dict[str, Any]] = {}
    for component, module in modules.items():
        params = 0
        dtype = None
        for key, value in module.state_dict().items():
            t = value.detach().cpu().contiguous()
            tensors[f"{component}.{key}"] = t
            params += t.numel()
            dtype = str(t.dtype).replace("torch.", "")
        tensor_meta[component] = {"dtype": dtype, "num_params": params}

    _st_save(tensors, str(root / _WEIGHTS_FILE))

    (root / _CONFIG_FILE).write_text(json.dumps(model.config.to_dict(), indent=2))

    metadata = {
        "format_version": FORMAT_VERSION,
        "library_version": _library_version(),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_model": _base_model_info(model),
        "hats_present": present,
        "layer_attachment": _layer_attachment(model, present),
        "step_set": list(model.config.thinker.step_set),
        "tensor_format": "safetensors",
        "tensors": tensor_meta,
    }
    (root / _METADATA_FILE).write_text(json.dumps(metadata, indent=2))


def _read_json(root: Path, name: str) -> dict[str, Any]:
    p = root / name
    if not p.is_file():
        raise InvalidHatCheckpointError(f"Not a valid hat checkpoint: missing {name}.")
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise InvalidHatCheckpointError(
            f"Not a valid hat checkpoint: {name} is not valid JSON."
        ) from e


def _require(meta: dict[str, Any], key: str) -> Any:
    if key not in meta:
        raise InvalidHatCheckpointError(
            f"Not a valid hat checkpoint: missing metadata key '{key}'."
        )
    return meta[key]


def _validate(model: "HatEnabledModel", meta: dict[str, Any]) -> None:
    fmt = _require(meta, "format_version")
    if fmt > FORMAT_VERSION:
        raise IncompatibleHatCheckpointError(
            f"Checkpoint format_version {fmt} is newer than this library "
            f"supports (max {FORMAT_VERSION}). Upgrade hugging-hat."
        )

    ckpt_hidden = _require(meta, "base_model")["hidden_size"]
    if ckpt_hidden != model.hidden_size:
        raise IncompatibleHatCheckpointError(
            f"hidden_size mismatch: checkpoint trained for {ckpt_hidden}, "
            f"target model has {model.hidden_size}. Hats are not transferable "
            f"across hidden sizes."
        )

    present = _require(meta, "hats_present")
    available = _hat_modules(model)
    for hat in present:
        if hat not in available:
            raise IncompatibleHatCheckpointError(
                f"Checkpoint contains '{hat}' weights but config.{hat}.enabled "
                f"is False. Enable it or load a matching checkpoint."
            )

    attachment = _require(meta, "layer_attachment")
    for hat in present:
        idx = attachment.get(hat, {}).get("resolved_index")
        if idx is not None and idx >= model.num_layers:
            raise IncompatibleHatCheckpointError(
                f"'{hat}' attaches at block {idx} but target model has only "
                f"{model.num_layers} blocks."
            )

    if "router" in present:
        ckpt_steps = len(_require(meta, "step_set"))
        target_steps = len(model.config.router.step_set)
        if ckpt_steps != target_steps:
            raise IncompatibleHatCheckpointError(
                f"router step_set size mismatch: checkpoint {ckpt_steps}, "
                f"config {target_steps}. The router classifier shape depends "
                f"on step_set."
            )


def _soft_checks(model: "HatEnabledModel", meta: dict[str, Any]) -> None:
    running = _library_version()
    ckpt_version = meta.get("library_version")
    if ckpt_version and ckpt_version != running:
        warnings.warn(
            f"Checkpoint written by hugging-hat {ckpt_version}, running "
            f"{running}. Behavior may differ.",
            UserWarning,
            stacklevel=3,
        )

    ckpt_base = meta.get("base_model", {})
    target = _base_model_info(model)
    if ckpt_base.get("name_or_path") != target["name_or_path"] or ckpt_base.get(
        "model_type"
    ) != target["model_type"]:
        warnings.warn(
            f"Loading hats trained on '{ckpt_base.get('name_or_path')}' onto "
            f"'{target['name_or_path']}'. Architecturally compatible but not "
            f"verified - outputs may differ.",
            UserWarning,
            stacklevel=3,
        )


def load_hats(model: "HatEnabledModel", path: str) -> None:
    root = Path(path)
    weights_path = root / _WEIGHTS_FILE
    if not weights_path.is_file():
        raise InvalidHatCheckpointError(
            f"Not a valid hat checkpoint: missing {_WEIGHTS_FILE}."
        )

    metadata = _read_json(root, _METADATA_FILE)
    _validate(model, metadata)
    _soft_checks(model, metadata)

    flat = _st_load(str(weights_path))

    modules = _hat_modules(model)
    for component in metadata["hats_present"]:
        module = modules[component]
        prefix = f"{component}."
        sub = {k[len(prefix):]: v for k, v in flat.items() if k.startswith(prefix)}
        try:
            module.load_state_dict(sub)
        except RuntimeError as e:
            raise IncompatibleHatCheckpointError(
                f"Shape mismatch loading '{component}': {e}. "
                f"Check thinker.hidden_multiplier / critic.num_heads."
            ) from e
