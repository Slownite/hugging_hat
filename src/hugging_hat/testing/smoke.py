from __future__ import annotations

import sys


def run_smoke() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("Skipping smoke: torch not installed. Install with `pip install -e '.[torch]'`.")
        return

    from hugging_hat.config import HatConfig
    from hugging_hat.model import HatEnabledModel
    from hugging_hat.testing.dummy_hf import build_dummy_causallm

    base = build_dummy_causallm(hidden_size=32, num_hidden_layers=8)
    hats = HatEnabledModel(
        base,
        config=HatConfig(
            layers_path="model.layers",
        ),
    )

    input_ids = torch.randint(0, base.config.vocab_size, (2, 16))
    attention_mask = torch.ones_like(input_ids)

    out = hats(input_ids=input_ids, attention_mask=attention_mask)
    assert isinstance(out, dict) and "logits" in out

    hats.set_steps_override(4)
    generated = hats.generate(input_ids=input_ids[:, :4], attention_mask=attention_mask[:, :4], max_new_tokens=3)
    assert generated.shape[1] == 7
    hats.clear_steps_override()


if __name__ == "__main__":
    run_smoke()
    sys.exit(0)
