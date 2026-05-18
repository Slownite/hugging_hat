# Hat Checkpoint Format (v0)

Spec for the on-disk **hat-only** artifact: what is written, how, and the rules
for safely reloading hats onto a compatible Base Model.

Tracks issue #2 (child of PRD #1). This document is the contract that
`save_hats(path)` / `load_hats(path)` (separate implementation issue) must
satisfy. It defines behavior, not code.

## Goals & non-goals

- **Save only hat parameters + metadata.** Never write the Base Model
  checkpoint in the default path (PRD: "Out of Scope" / decisions).
- **Small and shareable** (PRD user story 7), **reloadable onto a fresh
  instance of the same Base Model class** (story 8), **version-aware** (story
  23), and carrying enough provenance to detect incompatibilities (story 22).
- Non-goal: a general HF `save_pretrained` replacement, multi-format support,
  or backward-migration tooling. v0 ships exactly one format.

## 1. Artifact layout

A hat checkpoint is a **directory** (mirrors the HF `save_pretrained`
convention users already expect, and keeps the format extensible):

```
<path>/
  hats.safetensors      # all hat parameter tensors
  hat_config.json       # serialized HatConfig (to reconstruct HatEnabledModel)
  hat_metadata.json     # provenance + compatibility metadata
```

Rationale for splitting config vs metadata:

- `hat_config.json` is exactly `HatConfig.to_dict()` — it is *input* used to
  re-create the wrapper, and is round-trippable via `hat_config_from_dict`.
- `hat_metadata.json` is *provenance describing the trained artifact* (what it
  was trained against). It is read for compatibility checks, never fed back
  into `HatConfig`.

A single-file form is explicitly **not** offered in v0 (keeps the loader
simple; directory leaves room for future per-hat shards).

## 2. Tensor serialization format

**Decision: `safetensors`.** Default and only format in v0.

Why over `torch.save`:

- No pickle / arbitrary-code-execution risk — artifacts are meant to be
  *shared* (PRD story 7).
- Stable, framework-neutral, zero-copy, and already the HF ecosystem default.
- Deterministic byte layout aids the save/load round-trip test (PRD Testing
  Decisions).

Consequence: `safetensors` is added to the existing **`[torch]`** optional
extra (anyone who can run hats can load them). `torch.save` is **not**
supported as a fallback in v0 — one format only.

### Tensor key naming

All hat tensors live in the single `hats.safetensors`, keyed by
`"<component>.<module_state_dict_key>"`:

| Component | Prefix     | Present when                |
|-----------|------------|-----------------------------|
| Thinker   | `thinker.` | `config.thinker.enabled`    |
| Router    | `router.`  | `config.router.enabled`     |
| Critic    | `critic.`  | `config.critic.enabled`     |

Example keys: `thinker.norm.weight`, `thinker.mlp.0.weight`,
`thinker.mlp.2.bias`, `thinker.gate.weight`, `router.classifier.weight`,
`critic.attn.in_proj_weight`, `critic.gate.bias`.

Only instantiated hats are written; the set of present prefixes is also
recorded in metadata (`hats_present`) so a partial artifact (e.g. Thinker-only,
the v0 path) is unambiguous.

All tensors are saved in their **trained dtype** (no implicit cast); the dtype
per component is recorded in metadata.

## 3. Required metadata keys (`hat_metadata.json`)

```jsonc
{
  "format_version": 1,                 // int; THIS spec's schema version
  "library_version": "0.1.0",          // hugging_hat package version
  "created_at": "2026-05-18T12:00:00Z",// ISO-8601 UTC
  "base_model": {
    "name_or_path": "Qwen/Qwen2.5-0.5B",
    "model_type": "qwen2",             // HF base_model.config.model_type
    "hidden_size": 896,
    "num_hidden_layers": 24
  },
  "hats_present": ["thinker"],         // subset of thinker|router|critic
  "layer_attachment": {                // selector -> resolved concrete index
    "thinker": { "selector": "mid", "resolved_index": 12 }
  },
  "step_set": [0, 2, 4, 8],            // thinker.step_set (router uses same)
  "tensor_format": "safetensors",
  "tensors": {                         // per-component, for fail-fast checks
    "thinker": { "dtype": "float32", "num_params": 4724736 }
  }
}
```

Field requirements:

- `format_version`, `library_version`, `created_at`, `base_model.hidden_size`,
  `base_model.num_hidden_layers`, `hats_present`, `layer_attachment`,
  `step_set`, `tensor_format` are **required**.
- `base_model.name_or_path`, `base_model.model_type` are required but used only
  for *soft* checks (see §4).
- `tensors` is required and drives fast shape/param sanity before the full load.

## 4. Compatibility rules & failure modes

Checked by `load_hats()` in this order. **Hard** = raise, refuse to load.
**Soft** = `warnings.warn`, continue.

### Hard errors (refuse load)

| Condition | Exception | Message |
|---|---|---|
| `format_version` > reader's max supported | `IncompatibleHatCheckpointError` | `Checkpoint format_version {got} is newer than this library supports (max {max}). Upgrade hugging-hat.` |
| `base_model.hidden_size` ≠ target `hidden_size` | `IncompatibleHatCheckpointError` | `hidden_size mismatch: checkpoint trained for {ckpt}, target model has {target}. Hats are not transferable across hidden sizes.` |
| A `hats_present` component is disabled in target `HatConfig` | `IncompatibleHatCheckpointError` | `Checkpoint contains '{hat}' weights but config.{hat}.enabled is False. Enable it or load a matching checkpoint.` |
| Resolved attach index for a present hat ≥ target `num_hidden_layers` | `IncompatibleHatCheckpointError` | `'{hat}' attaches at block {idx} but target model has only {n} blocks.` |
| `router` present and `len(step_set)` ≠ target `router.step_set` length | `IncompatibleHatCheckpointError` | `router step_set size mismatch: checkpoint {a}, config {b}. The router classifier shape depends on step_set.` |
| Tensor shape mismatch on load (covers `thinker.hidden_multiplier`, `critic.num_heads`, any residual mismatch) | `IncompatibleHatCheckpointError` | `Shape mismatch loading '{key}': checkpoint {shape_a} vs model {shape_b}. Check thinker.hidden_multiplier / critic.num_heads.` |
| Missing required file or required metadata key | `InvalidHatCheckpointError` | `Not a valid hat checkpoint: missing {file_or_key}.` |

### Soft warnings (load proceeds)

| Condition | Message |
|---|---|
| `base_model.name_or_path` / `model_type` differs from target | `Loading hats trained on '{ckpt}' onto '{target}'. Architecturally compatible but not verified — outputs may differ.` |
| `library_version` differs from running version | `Checkpoint written by hugging-hat {a}, running {b}. Behavior may differ.` |
| `created_at` unparseable / absent `tensors` per-component entry | `Checkpoint metadata incomplete ({what}); skipping that pre-check.` |

### Round-trip guarantee

For a fixed seed and fixed inputs, `save_hats` then `load_hats` onto a fresh
instance of the same Base Model class must yield **bit-identical** model
outputs (PRD Testing Decisions). This is the primary acceptance test for the
implementation issue.

## Resolved decisions

- **`safetensors` dependency**: added to the existing `[torch]` extra.
- **Resume-training state** (optimizer/scheduler, PRD story 26): **out of
  scope** for this format. It will be a separate artifact owned by a later
  training issue; this format stays strictly hat-weights + metadata.
