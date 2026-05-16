# Hugging Hat

A modern Python project built using `uv`, Nix flakes, and `click`.

## What this is

`hugging-hat` is an open-source library to inject **test-time compute** (“thinking loops”) into *any* decoder-only Hugging Face causal LM via lightweight trainable modules (“Hats”) attached to a frozen base model.

Core idea: instead of scaling the base model at training time, we add **depth-in-time** at inference by iterating a tiny, parameter-shared module on hidden states.

## Hats

- **Thinker Hat (latent recurser)**: post-block module that loops on `(B,S,H)` hidden states for `num_steps` iterations using a gated residual MLP + RMSNorm.
- **Latent Router Hat (traffic controller)**: early-layer classifier that picks `num_steps` from a discrete step set (default `{0,2,4,8}`) using per-sequence mean pooling; decision is **prefill-fixed** during `.generate()`.
- **Cross-Attentive Critic Hat (auditor)**: late-layer cross-attention that compares current hidden states against cached early-layer “prompt memory” to dampen inconsistencies (disabled by default in v0).

## File structure (v0)

```
src/hugging_hat/
  config.py          # HatConfig dataclasses + YAML IO
  hf.py              # HF layer-stack resolution utilities
  model.py           # HatEnabledModel wrapper + hook management
  hats/
    thinker.py       # ThinkerHat (gated residual, parameter-shared)
    router.py        # LatentRouterHat (discrete step selection)
    critic.py        # CrossAttentiveCriticHat (cross-attn to prompt memory)
    norm.py          # RMSNorm
  cli.py             # minimal CLI (hh doctor)
```

## Requirements

- [Nix](https://nixos.org/download.html) (with flakes enabled)
- [direnv](https://direnv.net/) (optional, but highly recommended)

## Getting Started

1. **Load the environment**:
   If you have `direnv` installed, run:
   ```bash
   direnv allow
   ```
   Otherwise, enter the Nix shell manually:
   ```bash
   nix develop
   ```

2. **Install dependencies**:
   Inside the environment, install the project dependencies using `just`:
   ```bash
   just install
   ```
   This uses `uv` under the hood to create the virtual environment and install dependencies.

## Install (library)

Base install:
```bash
pip install -e .
```

Extras:
```bash
pip install -e ".[hf,torch,yaml]"
```

3. **Run the CLI**:
   The CLI entrypoint is `hh`. You can run it via `just run`:
   ```bash
   just run hello
   ```

## Development Commands

We use `just` as our command runner. Run `just` without arguments to see all available commands.

- `just install`: Install the package and dependencies
- `just run [args...]`: Run the CLI
- `just count-lines`: Count the lines of Python code in the `src` directory
- `just build`: Build the Python wheel/sdist

## Training strategy (v0)

Freeze the base model parameters; train only hats.

Recommended staged approach:
1) **Train Thinker** at a fixed `num_steps` (e.g., 4) on synthetic coding tasks with next-token CE.
2) **Train Router** with a teacher sweep over `{0,2,4,8}` using CE loss on reference continuations as the label target.
3) **Train Critic (optional)** last with small auxiliary consistency objectives + CE, and keep it opt-in by default.
