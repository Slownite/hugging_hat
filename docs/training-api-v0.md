# Training API (v0)

Shared reference for the v0 Thinker-Hat training surface. Issues #7–#14 (children
of PRD #1) each implement a slice of this; they all refer back here so the API
stays consistent. This document defines **intended shape**, not final code — an
implementer may refine signatures, but should keep the public surface small and
the boundaries below intact.

Grounding: the library already provides the wrapper
([model.py](model.py)), config ([config.py](config.py)), JSONL loader
([data.py](data.py)), prompt-masked preprocessor
([tokenizer.py](tokenizer.py)), and hat-only persistence
([persistence.py](persistence.py)). v0 training composes these — it does **not**
re-implement them.

## Boundaries (what owns what)

| Concern | Owner | Status |
|---|---|---|
| Hook wiring, frozen base forward, `save_hats`/`load_hats` | `HatEnabledModel` | done (#2–#4) |
| JSONL → `PromptCompletion` records | `data.load_jsonl` | done (#5) |
| `PromptCompletion` → `{input_ids, attention_mask, labels}` | `tokenizer.preprocess_record` | done (#6) |
| HF dataset → `PromptCompletion` records | `data` (new adapter) | #13 |
| Collate a batch of preprocessed records into padded tensors | `train.collate` | #7 |
| One forward/backward/step on hats only | `train.training_step` | #8 |
| Device + dtype/autocast policy | `train.device` | #9 |
| Loss/throughput logging + seeding | `train.logging` / `train.seed` | #10 |
| Resume from a saved hat checkpoint | `train` + persistence | #11 |
| Orchestration loop (`train_thinker`) | `train.train_thinker` | #8/#12 |
| `hh train-thinker` CLI (JSONL) | `cli` | #12 |
| `hh train-thinker` CLI (HF) | `cli` | #14 |

New code lives in a new package `src/hugging_hat/train/` (or a single
`src/hugging_hat/train.py` if it stays small). CLI additions go in
[cli.py](cli.py). A new optional extra is **not** needed — `train` already
exists in `pyproject.toml` (`accelerate`, `datasets`, `tqdm`).

## Proposed public surface

```python
# hugging_hat/train/__init__.py  (names, not necessarily one module)

@dataclass(frozen=True)
class TrainConfig:
    # data
    max_length: int = 1024
    batch_size: int = 1
    # optim
    lr: float = 1e-4
    weight_decay: float = 0.0
    max_steps: int | None = None         # cap by optimizer steps
    num_epochs: int = 1                  # used when max_steps is None
    grad_accum_steps: int = 1
    grad_clip: float | None = 1.0
    # thinker training
    thinker_steps: int = 4               # FIXED think-step count during training
    # runtime
    device: str | None = None            # None -> auto (cuda else cpu); #9
    precision: Literal["fp32", "fp16"] = "fp32"  # fp16 -> autocast+GradScaler; #9
    seed: int | None = None              # #10
    # logging
    log_every: int = 10                  # optimizer steps; #10
    # checkpointing
    save_every: int | None = None        # steps; None -> only at end
    resume_from: str | None = None       # path to a hat checkpoint dir; #11 (Option A: weights only)


@dataclass
class TrainResult:
    steps: int
    final_loss: float
    checkpoint_path: str | None


def train_thinker(
    model: HatEnabledModel,
    records: Iterable[PromptCompletion],
    tokenizer: Any,
    config: TrainConfig,
    *,
    output_dir: str,
) -> TrainResult: ...
```

Helper functions the loop is built from (each its own issue):

```python
def freeze_base_enable_hats(model: HatEnabledModel) -> list[nn.Parameter]: ...   # #8
def collate(batch: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, Tensor]: ...  # #7
def training_step(model, batch, optimizer, *, scaler=None, thinker_steps: int) -> StepMetrics: ...  # #8
def resolve_device(requested: str | None) -> torch.device: ...  # #9
def set_seed(seed: int | None) -> None: ...  # #10
```

## Key contracts

- **Fixed think-steps during training.** The loop sets a fixed step count via
  `model.set_steps_override(config.thinker_steps)` so the Thinker always runs
  `thinker_steps` iterations regardless of the (untrained) router. Cleared on
  exit. This is the v0 Thinker-first contract (PRD stories 4, 19; ARCHITECTURE
  staging step 1).
- **Frozen base.** `freeze_base_enable_hats` sets `requires_grad=False` on every
  base-model param and `True` on hat params, and returns only the trainable hat
  params for the optimizer. Observable test: base params get no grad/update; hat
  params do (PRD Testing Decisions).
- **Loss.** `labels` from the preprocessor already carry `IGNORE_INDEX` on the
  prompt span; pass them straight to the base model's CE (the HF model computes
  loss when `labels` is provided, using `ignore_index=-100`). The loop does not
  re-implement CE.
- **Hat-only checkpoints.** Saving uses `model.save_hats(output_dir)`. The base
  checkpoint is never written. Resume reads hats via `model.load_hats(...)`;
  optimizer/scheduler/step-counter state is **not** persisted — see the
  [Resume](#resume) section for the rationale (#11).

## Batching

v0 uses **dynamic per-batch padding** (option A in #7). Each batch is padded to
the longest sequence *in that batch*; nothing is packed across examples and no
batch is padded all the way to `max_length`.

### Decision summary

| Aspect | v0 choice |
|---|---|
| Strategy | Dynamic padding per batch (no packing) |
| Truncation | Stays in `tokenizer.preprocess_record`; `collate` never re-truncates |
| Padding side | Right (HF causal-LM convention) |
| `pad_token_id` fallback | If `tokenizer.pad_token_id is None`, fall back to `tokenizer.eos_token_id` and emit a one-time `warnings.warn` |
| Pad value — `input_ids` | `pad_token_id` |
| Pad value — `attention_mask` | `0` |
| Pad value — `labels` | `-100` (`IGNORE_INDEX`) so padded positions never contribute to loss |
| Output dtype | `torch.long` for all three tensors |

Rationale: option A is the minimum viable choice that respects the 1080 Ti OOM
story (PRD story 11) without wasting compute on uniformly `max_length`-padded
batches, and it composes cleanly with the per-record truncation the preprocessor
already does. Options B and C are explicitly deferred (see "Out of scope"
below).

### `collate` contract

```python
def collate(
    batch: list[dict[str, list[int]]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]: ...
```

- **Input.** A non-empty list of preprocessed records as emitted by
  `tokenizer.preprocess_record` — each a dict with keys `input_ids`,
  `attention_mask`, `labels` whose values are `list[int]` of the same length
  *within* a record but variable length *across* records. Records are assumed
  already truncated to `max_length`; `collate` does not verify this and does
  not re-truncate.
- **Output.** A dict with the same three keys, each a `torch.Tensor` of shape
  `(B, T)` and dtype `torch.long`, where `B = len(batch)` and `T` is the max
  per-record length in this batch. Right-padded with the pad values in the
  table above.
- **`pad_token_id`.** Required positional argument. Callers (training loop,
  CLI) are responsible for resolving the fallback to `eos_token_id` and
  emitting the warning before constructing the dataloader; `collate` itself
  trusts the value it receives. This keeps `collate` free of tokenizer state.
- **No device move.** `collate` returns CPU tensors. Device placement is #9's
  responsibility.

### Surface impact

- **Python API.** `TrainConfig.batch_size` is the dataloader batch size; it
  controls `B` directly. `TrainConfig.max_length` is passed through to the
  preprocessor (the per-record truncation cap) and is *not* enforced again by
  `collate`. The loop wires `collate` into the `DataLoader` via
  `functools.partial(collate, pad_token_id=resolved_pad_id)`.
- **CLI (`hh train-thinker`).** `--batch-size` and `--max-length` map 1:1 onto
  the `TrainConfig` fields above. If the tokenizer has no pad token, the CLI
  prints a single warning naming the fallback (`eos_token_id`) so users see it
  even when Python warnings are silenced.

### Out of scope for v0 (deferred)

- **Option B — fixed-length padding to `max_length`.** Not adopted; would waste
  compute on ragged batches. Revisit only if a real OOM repro shows option A's
  variable-length batches blow memory budgets on a 1080 Ti.
- **Option C — example packing.** Not adopted; the cross-example attention
  masking and label-segmenting it requires is disproportionate for v0 throughput
  needs. Revisit once the loop is stable and a throughput target exists.
- **Left padding.** Not adopted for training. (Generation-time left padding is a
  separate concern and not in scope here.)

## Resume

v0 adopts **option A: hat-weights-only resume** (#11). `TrainConfig.resume_from`
points at a hat checkpoint directory; the loop calls `model.load_hats(path)`
before constructing the optimizer and continues with a **fresh optimizer,
scheduler, and step counter**. No sibling artifact is written.

### Decision summary

| Aspect | v0 choice |
|---|---|
| What is persisted for resume | Hat weights only (the existing `save_hats` artifact) |
| Optimizer state (Adam moments) | **Not** restored — intentionally reset |
| LR-scheduler state | **Not** restored — intentionally reset |
| Global step counter | **Not** restored — `TrainResult.steps` counts only the resumed run |
| RNG state | **Not** restored — `TrainConfig.seed` reseeds the run |
| Resume artifact location | N/A — no extra artifact for v0 |
| Resume artifact format | N/A |
| Invalid `resume_from` | Surfaces `persistence.InvalidHatCheckpointError` (or `IncompatibleHatCheckpointError` on shape/config mismatch) before training starts |

### Tradeoffs

- **Reproducibility.** The first few post-resume steps will not match an
  uninterrupted run: Adam's first/second moments and the LR-schedule position
  are gone. This is the deliberate cost of option A. Loss curves typically
  converge again within a small number of steps on overfit-style data.
- **Format simplicity.** No new on-disk format to version or maintain. The
  hat checkpoint stays the single shareable artifact, exactly as specified in
  [hat-checkpoint-format.md](hat-checkpoint-format.md).
- **Pickle-safety.** Avoids introducing a `torch.save`/pickle surface for
  optimizer state — keeping resume artifacts safe to share without the
  pickle-deserialization caveat the checkpoint format was designed to avoid.

Faithful-continuation resume (option B: optimizer + scheduler + RNG state in a
sibling `trainer_state.pt`) is **deferred**. Revisit only when a use case
demands bit-exact resume — long pre-training runs, paper reproducibility, or a
scheduler whose mid-run position matters (e.g. warmup-cosine across millions of
steps).

### Surface

- **Python API.** `TrainConfig.resume_from: str | None = None` — path to a hat
  checkpoint directory. `train_thinker` calls `model.load_hats(resume_from)`
  before freezing/optimizer setup. `TrainResult.steps` reflects this run only.
- **CLI (`hh train-thinker`, #12).** `--resume-from PATH` maps 1:1 onto
  `TrainConfig.resume_from`.

## Sequencing

```
#7 collate ─┐
#9 device ──┤
#10 seed ───┼─> #8 training_step + train_thinker ─> #12 CLI (JSONL) ─┐
            │                                                         ├─> done
#13 HF fields decision ─> (HF adapter) ─────────────> #14 CLI (HF) ──┘
#11 resume decision ─> resume support (folds into #8/#12)
```
