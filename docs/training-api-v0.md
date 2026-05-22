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
    resume_from: str | None = None       # path to a hat checkpoint dir; #11


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
  whether optimizer/scheduler state is also persisted is the open decision in
  #11.

## Sequencing

```
#7 collate ─┐
#9 device ──┤
#10 seed ───┼─> #8 training_step + train_thinker ─> #12 CLI (JSONL) ─┐
            │                                                         ├─> done
#13 HF fields decision ─> (HF adapter) ─────────────> #14 CLI (HF) ──┘
#11 resume decision ─> resume support (folds into #8/#12)
```
