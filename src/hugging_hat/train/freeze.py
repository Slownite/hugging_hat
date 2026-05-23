from __future__ import annotations

from torch import nn

from hugging_hat.model import HatEnabledModel

_HAT_PREFIXES = ("thinker_hat.", "router_hat.", "critic_hat.")


def freeze_base_enable_hats(model: HatEnabledModel) -> list[nn.Parameter]:
    """Freeze base-model params, unfreeze hat params, return the hat params.

    The returned list is what the optimizer should be constructed over.
    """
    hat_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if name.startswith(_HAT_PREFIXES):
            param.requires_grad = True
            hat_params.append(param)
        else:
            param.requires_grad = False
    if not hat_params:
        raise ValueError(
            "freeze_base_enable_hats: model has no hat parameters to train. "
            "Enable at least one of thinker/router/critic in HatConfig."
        )
    return hat_params
