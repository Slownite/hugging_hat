from __future__ import annotations

import torch

from hugging_hat.tokenizer import IGNORE_INDEX


def collate(
    batch: list[dict[str, list[int]]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Right-pad a list of preprocessed records into (B, T) long tensors.

    Contract is documented in ``docs/training-api-v0.md`` under "Batching".
    Records are assumed already truncated to ``max_length`` by the preprocessor;
    this function never re-truncates and never moves tensors to a device.
    """
    if not batch:
        raise ValueError("collate received an empty batch")

    lengths = [len(rec["input_ids"]) for rec in batch]
    target_len = max(lengths)

    input_ids = torch.full((len(batch), target_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), target_len), dtype=torch.long)
    labels = torch.full((len(batch), target_len), IGNORE_INDEX, dtype=torch.long)

    for i, rec in enumerate(batch):
        n = lengths[i]
        input_ids[i, :n] = torch.tensor(rec["input_ids"], dtype=torch.long)
        attention_mask[i, :n] = torch.tensor(rec["attention_mask"], dtype=torch.long)
        labels[i, :n] = torch.tensor(rec["labels"], dtype=torch.long)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
