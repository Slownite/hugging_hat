## Problem Statement

As a developer building and experimenting with **Hats** (lightweight trainable modules) on top of a frozen **Base Model**, I need a simple and repeatable way to train the **Thinker Hat** first (then the **Latent Router Hat**, then the **Cross-Attentive Critic Hat**) without downloading or modifying base-model weights, and without forcing users to abandon normal Hugging Face `.generate()` workflows.

Today, the library provides hat modules and a `HatEnabledModel` wrapper, but it does not yet provide an “easy mode” training experience via CLI or a small Python API. Users need a clear path to:

- prepare data (JSONL prompt/completion or Hugging Face datasets),
- run a hat-only training loop on a single GPU (1080 Ti) or CPU fallback,
- save/load only hat weights,
- evaluate and iterate quickly.

## Solution

Provide a v0 training experience focused on **Thinker Hat** training that is usable both:

1) via CLI (minimal, batteries-included commands), and
2) via a small Python training API (callable from scripts/notebooks),

with these user-visible properties:

- **Base Model** parameters are frozen; only hat parameters update.
- Training uses standard next-token cross entropy on concatenated prompt+completion, with the prompt portion masked out.
- Inputs support both **JSONL prompt/completion** and **Hugging Face datasets** from day one.
- Artifacts saved are **only hat weights** plus metadata (no full base checkpoint).
- Training is designed for **single GPU + CPU fallback**, optimized for practicality on a 1080 Ti (cu118), not multi-GPU.

## User Stories

1. As a library user, I want to train a **Thinker Hat** on my own JSONL prompt/completion file, so that I can improve coding performance without retraining the base model.
2. As a library user, I want to train a **Thinker Hat** using a Hugging Face dataset identifier and split, so that I can start quickly without building data tooling.
3. As a library user, I want the training loop to freeze the **Base Model** automatically, so that I don’t accidentally update billions of parameters.
4. As a library user, I want to choose a fixed Think Step count during Thinker training, so that training is stable and reproducible.
5. As a library user, I want the training code to support fp16 mixed precision on a single GPU, so that training is feasible on consumer hardware.
6. As a library user, I want a CPU fallback mode, so that I can run sanity checks without a GPU.
7. As a library user, I want to save only the trained hat weights, so that artifacts are small and shareable.
8. As a library user, I want to load hat weights onto a new instance of the same Base Model class, so that I can reuse trained hats in inference.
9. As a library user, I want an easy way to override steps at inference time, so that I can do ablations (e.g., force `0` vs `4` steps).
10. As a library user, I want training logs that include loss, tokens/sec, and step count, so that I can track progress.
11. As a library user, I want to cap max sequence length and batch size, so that I can avoid OOM errors.
12. As a library user, I want to pack or pad examples predictably, so that training throughput is stable.
13. As a library user, I want to configure which block index the **Thinker Hat** attaches to (semantic-first with index overrides), so that I can reproduce experiments across different base models.
14. As a library user, I want the CLI to print a clear summary of resolved attach points and step set, so that I can verify configuration before training.
15. As a library user, I want to run a tiny smoke training job end-to-end, so that I can validate my environment.
16. As a library user, I want the training code to work without requiring a model download at CLI time (when I pass a local path), so that I can train offline.
17. As a library user, I want training to be deterministic when I set a seed, so that I can compare experiments.
18. As a library user, I want the library to avoid forcing specific PyTorch wheels, so that I can choose cu118 wheels appropriate for a 1080 Ti.
19. As a library user, I want to select which Hat components are enabled during training, so that Thinker-only training remains clean.
20. As a library user, I want minimal defaults that “just work” for coding tasks, so that I don’t need to tune every knob.
21. As a library user, I want the training API to be callable from Python with a small surface area, so that I can embed it into my own experiment runner.
22. As a library user, I want training runs to emit a metadata file capturing the Base Model identifier, hidden size, layer selectors, and step set, so that I can load compatible hats later.
23. As a library user, I want the saved hat artifact to include version info, so that I can detect incompatibilities across library versions.
24. As a library user, I want the data loader to validate schema early (required fields, types), so that I fail fast.
25. As a library user, I want the CLI to support both JSONL and HF datasets with a single command, so that I can switch sources easily.
26. As a library user, I want to resume training from a saved hat checkpoint, so that I can iterate without restarting.
27. As a library user, I want a clear path to training the **Latent Router Hat** next (teacher sweep), so that compute can be dynamic.
28. As a library user, I want a clear path to training the **Cross-Attentive Critic Hat** last, so that I can optionally improve consistency.

## Implementation Decisions

- Provide both CLI and Python API; CLI is a thin wrapper around the Python API.
- v0 training focuses on **Thinker Hat** first; **Router** and **Critic** training are explicitly staged follow-ups.
- Data sources supported from day one:
  - JSONL files containing prompt+completion pairs.
  - Hugging Face datasets identified by name/config/split.
- Loss computation:
  - Concatenate prompt and completion into a single sequence.
  - Compute next-token cross entropy with prompt tokens masked (labels set to ignore index).
- Hardware support:
  - Single-GPU focus with CPU fallback; no multi-GPU orchestration in v0.
  - Mixed precision support is allowed (especially fp16); compute dtype inside hats remains configurable.
- Checkpointing:
  - Save and load **only hat parameters** plus minimal metadata; never save the full base checkpoint in the default path.
- Wrapper integration:
  - Training uses `HatEnabledModel` so hats are applied via hooks while training calls remain “normal forward” usage.
  - The base model remains frozen by default; hats are trainable.
- CLI surface (v0 scope):
  - Add a `train-thinker` command that can consume either JSONL or HF datasets.
  - Print resolved attach indices (semantic selectors → concrete indices) and step set before training starts.
- Interfaces:
  - Add `save_hats(path)` / `load_hats(path)` on the wrapper as the primary user-facing persistence API.
  - Keep configuration config-first via `HatConfig`, with optional YAML IO.

## Testing Decisions

- Good tests focus on observable behavior:
  - Base model parameters remain frozen (no grads / no optimizer updates).
  - Hat parameters receive gradients and update.
  - Data masking works (loss only reflects completion tokens).
  - Save/load round-trips hat weights and restores identical outputs for a fixed seed and fixed inputs.
- Modules to test:
  - Data loading and schema validation for JSONL inputs.
  - HF dataset adapter (basic smoke; does not require large downloads in CI).
  - Training step function (one forward/backward/optimizer step updates only hat params).
  - Persistence layer for hat-only checkpoints + metadata.
- Prior art:
  - Reuse existing local smoke patterns (dummy model harness) to test hook wiring without real HF downloads where feasible.

## Out of Scope

- Multi-GPU distributed training.
- Full reproduction of Hugging Face `Trainer` feature set (callbacks, complex schedulers, DeepSpeed, etc.).
- Router teacher sweep implementation (stage 2) and critic training (stage 3) beyond documenting the intended path.
- Test-execution-based coding evaluation (unit tests, sandboxed compilation) as a training signal; v0 uses CE loss.

## Further Notes

- Compatibility constraints for a 1080 Ti strongly suggest cu118 wheels; the training UX should avoid surprising users by hard-pinning wheels in the library itself.
- Staging is intentional: Thinker-first creates immediate value and establishes a stable foundation for Router and Critic training.

