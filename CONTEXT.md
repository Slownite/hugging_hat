# Hugging Hat

An open-source Python library for adding configurable test-time compute to decoder-only LLMs via lightweight, trainable modules (“Hats”) attached to a frozen base model.

## Language

**Hat**:
A small trainable module added to a frozen base LLM to change its inference-time behavior.
_Avoid_: plugin, patch, adapter (unless specifically meant)

**Base Model**:
The underlying decoder-only LLM whose original parameters remain frozen.
_Avoid_: backbone, trunk

**Thinker Hat**:
A Hat that iteratively refines hidden states over multiple steps before the base model continues processing.
_Avoid_: latent loop, TRM (unless defined in the same document)

**Gated Residual Update**:
A Thinker Hat update rule that adds a gated delta to the current hidden states.
_Avoid_: always-on residual

**Per-token Gate**:
A gating scheme that produces one gate value per token position to modulate a Thinker Hat update.
_Avoid_: global gate

**Think Step**:
One iteration of the Thinker Hat applied to a hidden-state tensor.
_Avoid_: cycle, tick

**Latent Router Hat**:
A Hat that selects an inference-time compute budget (e.g., number of Think Steps) from early hidden states.
_Avoid_: controller, scheduler

**Step Set**:
The discrete set of allowable Think Step counts that the Latent Router Hat may choose from.
_Avoid_: ladder, schedule

**Per-sequence Routing**:
A routing mode where one compute-budget decision is made per input sequence (not per token position).
_Avoid_: per-token routing

**Prefill-fixed Routing**:
A routing mode where the compute budget is chosen once from the prompt prefill and reused during autoregressive decoding.
_Avoid_: per-token rerouting

**Post-block Attachment**:
A Hat placement where it receives the hidden states after a full transformer block and returns updated hidden states to feed into subsequent blocks.
_Avoid_: inside-layer, mid-block

## Relationships

- A **Hat** augments exactly one **Base Model**
- A **Thinker Hat** performs one or more **Think Steps**
- A **Latent Router Hat** selects a compute budget for a **Thinker Hat**
- A **Thinker Hat** uses **Post-block Attachment**

## Example dialogue

> **Dev:** “Where does the **Thinker Hat** run in the stack?”
> **Domain expert:** “Use **Post-block Attachment** so it works across different decoder-only architectures.”

## Flagged ambiguities

- “layer” vs “block” — resolved: “block” refers to one transformer block in the base model’s decoder stack.
