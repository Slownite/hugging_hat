from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .config import HatConfig, LayerSelector
from .hats import CrossAttentiveCriticHat, LatentRouterHat, ThinkerHat
from .hf import is_prefill_forward, resolve_decoder_layers


def _require_transformers() -> Any:
    try:
        import transformers  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Hugging Face integration requires optional dependency: `pip install hugging-hat[hf]`"
        ) from e
    return transformers


def _resolve_layer_index(selector: LayerSelector, *, num_layers: int) -> int:
    if isinstance(selector, int):
        idx = selector
    elif selector == "early":
        idx = min(2, num_layers - 1)
    elif selector == "mid":
        idx = num_layers // 2
    elif selector == "late":
        idx = max(num_layers - 2, 0)
    else:
        raise ValueError(f"Unknown layer selector: {selector!r}")
    if idx < 0 or idx >= num_layers:
        raise IndexError(f"Layer index {idx} out of range for num_layers={num_layers}")
    return idx


def _extract_hidden_states_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0]
    if hasattr(output, "hidden_states") and isinstance(output.hidden_states, torch.Tensor):
        return output.hidden_states
    raise TypeError("Block output did not contain a hidden-state tensor in a supported form")


def _replace_hidden_states_in_output(output: Any, new_hidden_states: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return new_hidden_states
    if isinstance(output, tuple):
        return (new_hidden_states, *output[1:])
    if isinstance(output, list):
        output = list(output)
        output[0] = new_hidden_states
        return output
    if hasattr(output, "hidden_states"):
        output.hidden_states = new_hidden_states
        return output
    raise TypeError("Cannot replace hidden states in unsupported output type")


@dataclass
class _EphemeralState:
    steps_override: int | None = None
    routed_steps: torch.Tensor | None = None  # (B,)
    prompt_memory: torch.Tensor | None = None  # (B,S,H)
    prompt_attention_mask: torch.Tensor | None = None  # (B,S)
    prefill_done: bool = False
    last_input_ids: torch.Tensor | None = None
    last_attention_mask: torch.Tensor | None = None
    last_past_key_values: Any | None = None
    last_is_prefill: bool = True


class HatEnabledModel(nn.Module):
    def __init__(self, base_model: nn.Module, *, config: HatConfig | None = None) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config or HatConfig()

        if not hasattr(base_model, "config"):
            raise TypeError("base_model must have a .config attribute (HF PreTrainedModel-like)")

        hidden_size = int(getattr(base_model.config, "hidden_size", 0) or getattr(base_model.config, "n_embd", 0))
        if hidden_size <= 0:
            raise ValueError("Could not infer hidden_size from base_model.config")
        self.hidden_size = hidden_size

        num_layers = int(
            getattr(base_model.config, "num_hidden_layers", 0) or getattr(base_model.config, "n_layer", 0)
        )
        if num_layers <= 0:
            layers, _ = resolve_decoder_layers(base_model, layers_path=self.config.layers_path)
            num_layers = len(layers)
        self.num_layers = num_layers

        self.layers, self.layers_path = resolve_decoder_layers(base_model, layers_path=self.config.layers_path)

        self.router_layer_idx = _resolve_layer_index(self.config.router.attach_layer, num_layers=self.num_layers)
        self.thinker_layer_idx = _resolve_layer_index(self.config.thinker.attach_layer, num_layers=self.num_layers)
        self.critic_layer_idx = _resolve_layer_index(self.config.critic.attach_layer, num_layers=self.num_layers)
        self.prompt_memory_layer_idx = _resolve_layer_index(
            self.config.critic.prompt_memory_layer, num_layers=self.num_layers
        )

        self.router_hat: LatentRouterHat | None = None
        if self.config.router.enabled:
            self.router_hat = LatentRouterHat(
                hidden_size=self.hidden_size,
                step_set=self.config.router.step_set,
                compute_dtype=self.config.router.compute_dtype,
            )

        self.thinker_hat: ThinkerHat | None = None
        if self.config.thinker.enabled:
            self.thinker_hat = ThinkerHat(
                hidden_size=self.hidden_size,
                hidden_multiplier=self.config.thinker.hidden_multiplier,
                use_rms_norm=self.config.thinker.use_rms_norm,
                compute_dtype=self.config.thinker.compute_dtype,
            )

        self.critic_hat: CrossAttentiveCriticHat | None = None
        if self.config.critic.enabled:
            num_heads = self.config.critic.num_heads
            if num_heads is None:
                num_heads = int(getattr(base_model.config, "num_attention_heads", 0) or 8)
            self.critic_hat = CrossAttentiveCriticHat(
                hidden_size=self.hidden_size,
                num_heads=num_heads,
                compute_dtype=self.config.critic.compute_dtype,
            )

        self._state = _EphemeralState()
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *, config: HatConfig | None = None, **kwargs: Any) -> "HatEnabledModel":
        transformers = _require_transformers()
        base_model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        return cls(base_model, config=config)

    def __getattr__(self, name: str) -> Any:
        if name in {"base_model", "config", "_state", "_hooks"}:
            return super().__getattribute__(name)
        return getattr(self.base_model, name)

    def set_steps_override(self, steps: int) -> None:
        self._state.steps_override = int(steps)

    def clear_steps_override(self) -> None:
        self._state.steps_override = None

    def clear_ephemeral_state(self) -> None:
        self._state = _EphemeralState(steps_override=self._state.steps_override)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _register_hooks(self) -> None:
        self.remove_hooks()

        self._hooks.append(self.base_model.register_forward_pre_hook(self._capture_model_inputs(), with_kwargs=True))

        if self.critic_hat is not None and self.config.critic.prompt_memory_source == "layer":
            self._hooks.append(self.layers[self.prompt_memory_layer_idx].register_forward_hook(self._prompt_memory_hook()))

        if self.router_hat is not None:
            self._hooks.append(self.layers[self.router_layer_idx].register_forward_hook(self._router_hook()))

        if self.thinker_hat is not None:
            self._hooks.append(self.layers[self.thinker_layer_idx].register_forward_hook(self._thinker_hook()))

        if self.critic_hat is not None:
            self._hooks.append(self.layers[self.critic_layer_idx].register_forward_hook(self._critic_hook()))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self.base_model.generate(*args, **kwargs)
        finally:
            self.clear_ephemeral_state()

    def _capture_model_inputs(self) -> Callable[[nn.Module, tuple[Any, ...], dict[str, Any]], tuple[tuple[Any, ...], dict[str, Any]]]:
        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
            input_ids = kwargs.get("input_ids")
            attention_mask = kwargs.get("attention_mask")
            past_key_values = kwargs.get("past_key_values")

            if input_ids is None and len(args) > 0 and isinstance(args[0], torch.Tensor):
                input_ids = args[0]
            if attention_mask is None and len(args) > 1 and isinstance(args[1], torch.Tensor):
                attention_mask = args[1]

            self._state.last_input_ids = input_ids
            self._state.last_attention_mask = attention_mask
            self._state.last_past_key_values = past_key_values
            self._state.last_is_prefill = is_prefill_forward(past_key_values=past_key_values, input_ids=input_ids)
            return args, kwargs

        return hook

    def _router_hook(self) -> Callable[[nn.Module, tuple[Any, ...], Any], Any]:
        def hook(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            if self.router_hat is None:
                return output

            hidden_states = _extract_hidden_states_from_output(output)
            attention_mask = self._state.last_attention_mask

            if self._state.prefill_done and self.config.router.prefill_fixed:
                return output

            router_out = self.router_hat(hidden_states, attention_mask=attention_mask)
            self._state.routed_steps = router_out.chosen_steps

            if self.config.critic.enabled and self.config.critic.prompt_memory_source == "router":
                if self._state.last_is_prefill:
                    self._state.prompt_memory = hidden_states.detach()
                    self._state.prompt_attention_mask = attention_mask.detach() if attention_mask is not None else None

            if self._state.last_is_prefill:
                self._state.prefill_done = True
            return output

        return hook

    def _thinker_hook(self) -> Callable[[nn.Module, tuple[Any, ...], Any], Any]:
        def hook(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            if self.thinker_hat is None:
                return output

            hidden_states = _extract_hidden_states_from_output(output)
            num_steps = self._resolve_steps_for_batch(hidden_states)
            if num_steps <= 0:
                return output

            updated, _ = self.thinker_hat(hidden_states, num_steps=num_steps)
            return _replace_hidden_states_in_output(output, updated)

        return hook

    def _critic_hook(self) -> Callable[[nn.Module, tuple[Any, ...], Any], Any]:
        def hook(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            if self.critic_hat is None:
                return output

            hidden_states = _extract_hidden_states_from_output(output)
            prompt_memory, prompt_mask = self._resolve_prompt_memory(inputs)
            if prompt_memory is None:
                return output

            updated, _ = self.critic_hat(hidden_states, prompt_memory=prompt_memory, attention_mask=prompt_mask)
            return _replace_hidden_states_in_output(output, updated)

        return hook

    def _prompt_memory_hook(self) -> Callable[[nn.Module, tuple[Any, ...], Any], Any]:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            if not self._state.last_is_prefill:
                return output
            hidden_states = _extract_hidden_states_from_output(output)
            self._state.prompt_memory = hidden_states.detach()
            if self._state.last_attention_mask is not None:
                self._state.prompt_attention_mask = self._state.last_attention_mask.detach()
            else:
                self._state.prompt_attention_mask = None
            return output

        return hook

    def _resolve_steps_for_batch(self, hidden_states: torch.Tensor) -> int:
        if self._state.steps_override is not None:
            return int(self._state.steps_override)

        if self._state.routed_steps is None:
            return int(self.config.thinker.default_steps)

        routed = self._state.routed_steps
        if routed.numel() == 0:
            return int(self.config.thinker.default_steps)
        return int(routed.max().item())

    def _resolve_prompt_memory(self, inputs: tuple[Any, ...]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.config.critic.prompt_memory_source == "router":
            return self._state.prompt_memory, self._state.prompt_attention_mask

        if self.config.critic.prompt_memory_source == "layer":
            return self._state.prompt_memory, self._state.prompt_attention_mask

        return None, None
