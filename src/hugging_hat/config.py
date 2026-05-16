from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ComputeDType = Literal["match_base", "float32"]
LayerSelector = int | Literal["early", "mid", "late"]


@dataclass(frozen=True, slots=True)
class ThinkerHatConfig:
    enabled: bool = True
    attach_layer: LayerSelector = "mid"
    step_set: tuple[int, ...] = (0, 2, 4, 8)
    default_steps: int = 0
    hidden_multiplier: int = 4
    use_rms_norm: bool = True
    compute_dtype: ComputeDType = "match_base"


@dataclass(frozen=True, slots=True)
class LatentRouterHatConfig:
    enabled: bool = True
    attach_layer: LayerSelector = "early"
    step_set: tuple[int, ...] = (0, 2, 4, 8)
    pooling: Literal["mean"] = "mean"
    prefill_fixed: bool = True
    compute_dtype: ComputeDType = "match_base"


@dataclass(frozen=True, slots=True)
class CrossAttentiveCriticHatConfig:
    enabled: bool = False
    attach_layer: LayerSelector = "late"
    prompt_memory_source: Literal["router", "layer"] = "router"
    prompt_memory_layer: LayerSelector = "early"
    num_heads: int | None = None
    compute_dtype: ComputeDType = "match_base"


@dataclass(frozen=True, slots=True)
class HatConfig:
    layers_path: str | None = None
    router: LatentRouterHatConfig = field(default_factory=LatentRouterHatConfig)
    thinker: ThinkerHatConfig = field(default_factory=ThinkerHatConfig)
    critic: CrossAttentiveCriticHatConfig = field(default_factory=CrossAttentiveCriticHatConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_yaml() -> Any:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "YAML support requires optional dependency: `pip install hugging-hat[yaml]`"
        ) from e
    return yaml


def hat_config_from_dict(data: dict[str, Any]) -> HatConfig:
    router = LatentRouterHatConfig(**data.get("router", {}))
    thinker = ThinkerHatConfig(**data.get("thinker", {}))
    critic = CrossAttentiveCriticHatConfig(**data.get("critic", {}))
    return HatConfig(layers_path=data.get("layers_path"), router=router, thinker=thinker, critic=critic)


def hat_config_from_yaml(text: str) -> HatConfig:
    yaml = _require_yaml()
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise TypeError("YAML root must be a mapping")
    return hat_config_from_dict(data)


def hat_config_to_yaml(config: HatConfig) -> str:
    yaml = _require_yaml()
    return yaml.safe_dump(config.to_dict(), sort_keys=False)

