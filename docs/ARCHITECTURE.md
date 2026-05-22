# Architecture (v0)

## Goal

Add configurable **test-time compute** to a frozen decoder-only causal LM by inserting small trainable modules (“Hats”) into the forward pass via PyTorch hooks, while keeping Hugging Face `.generate()` usable without custom generation code.

## Design overview

### Base model

- Any `transformers.AutoModelForCausalLM` instance.
- Base weights remain frozen during hat training.

### HatEnabledModel (wrapper)

`HatEnabledModel` is a thin `nn.Module` wrapper around a HF model instance:

- Resolves the decoder block stack (e.g. `model.layers`, `transformer.h`) via heuristics, with a `layers_path` override.
- Registers post-forward hooks on specific blocks (router/thinker/critic).
- Captures per-call inputs (`input_ids`, `attention_mask`, `past_key_values`) via a model-level forward pre-hook so block hooks have light context.
- Stores per-request ephemeral state (router decision, cached prompt memory) and clears it after `.generate()`.

### Hats

#### Latent Router Hat

- Attaches to an **early** block.
- Pools hidden states per sequence using **mean pooling over `attention_mask`**.
- Selects a step count from a discrete **step set** (default `{0,2,4,8}`).
- Decision is **prefill-fixed**: chosen once during prompt prefill and reused during autoregressive decode.

#### Thinker Hat

- Attaches **post-block** at a **mid** block.
- For `num_steps > 0`, applies `num_steps` iterations of a parameter-shared gated residual MLP update:
  - RMSNorm → MLP → sigmoid gate → residual add
- Uses a **per-token scalar gate** (shape `(B,S,1)`).
- `num_steps=0` is a true no-op (fast path).

#### Cross-Attentive Critic Hat (optional)

- Attaches to a **late** block (disabled by default).
- Cross-attends current hidden states to cached “prompt memory” captured from an early layer during prefill.
- Applies a gated residual update with the cross-attn output.

## Configuration

Configuration is config-first via dataclasses in `src/hugging_hat/config.py`, with optional YAML IO (`pip install hugging-hat[yaml]`).

## Persistence

Hats are saved/loaded **hat-only** (never the base checkpoint). The on-disk
format, metadata fields, and compatibility rules are specified in
[hat-checkpoint-format.md](hat-checkpoint-format.md).

## Training (staged)

The v0 training surface (Thinker-first) is specified in
[training-api-v0.md](training-api-v0.md).

1) Train **Thinker** at fixed steps (e.g. 4) with next-token CE on synthetic coding tasks.
2) Train **Router** using a teacher sweep over `{0,2,4,8}`: label each example with the step count that minimizes CE loss on the reference continuation.
3) Train **Critic** last (optional), with care to avoid suppressing useful creativity; keep opt-in by default.

