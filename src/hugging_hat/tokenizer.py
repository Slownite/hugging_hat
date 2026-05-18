from __future__ import annotations

from typing import Any

from hugging_hat.data import PromptCompletion

IGNORE_INDEX = -100


def preprocess_record(
    record: PromptCompletion,
    tokenizer: Any,
    max_length: int,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, list[int]]:
    """
    Preprocesses a prompt/completion record into model inputs and masked labels.

    This function concatenates the prompt and completion, tokenizes the full text,
    and masks out the prompt tokens in the labels array so the model only computes
    loss on the completion tokens.

    Truncation Policy:
    - If the combined text exceeds `max_length`, it is truncated to `max_length`.
    - If the prompt itself exceeds `max_length`, the completion is entirely lost
      and all labels will be masked with `ignore_index`.
    """
    # 1. Concatenate the prompt and completion
    full_text = record.prompt + record.completion

    # 2. Tokenize the full text
    tokenized_full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_special_tokens_mask=True,
    )

    # 3. Find out how many tokens the prompt span occupies in the full
    #    sequence. The prompt content is tokenized WITHOUT special tokens
    #    (since special tokens are added once, around the whole sequence),
    #    then offset by however many leading special tokens the tokenizer
    #    prepended to the full sequence (e.g. BOS/CLS). Without this offset
    #    the mask is off-by-one for prefix-adding tokenizers; counting the
    #    prompt's own special tokens instead would over-count for
    #    suffix-adding tokenizers (e.g. T5's trailing EOS).
    special_mask = tokenized_full["special_tokens_mask"]
    leading_special = 0
    for is_special in special_mask:
        if is_special:
            leading_special += 1
        else:
            break

    prompt_content = tokenizer(record.prompt, add_special_tokens=False)
    prompt_len = leading_special + len(prompt_content["input_ids"])

    # 4. Create the labels array (copy of input_ids)
    input_ids = tokenized_full["input_ids"]
    labels = list(input_ids)

    # 5. Mask the prompt tokens in the labels
    mask_length = min(prompt_len, len(labels))
    for i in range(mask_length):
        labels[i] = ignore_index

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized_full["attention_mask"],
        "labels": labels,
    }
