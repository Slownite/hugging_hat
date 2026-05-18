import pytest
from transformers import AutoTokenizer
from hugging_hat.data import PromptCompletion
from hugging_hat.tokenizer import preprocess_record, IGNORE_INDEX

@pytest.fixture(scope="module")
def tokenizer():
    """A small tokenizer for our tests. Scope is 'module' so it only loads once."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    # GPT-2 doesn't have a pad token by default, which can cause warnings when truncating
    tok.pad_token = tok.eos_token 
    return tok

def test_preprocess_record_masks_prompt_correctly(tokenizer):
    """Test that the prompt part of the sequence is masked with IGNORE_INDEX."""
    record = PromptCompletion(
        prompt="Translate to French: Hello",
        completion=" Bonjour"
    )
    
    result = preprocess_record(record, tokenizer, max_length=128)
    
    input_ids = result["input_ids"]
    labels = result["labels"]
    
    # Basic structural checks
    assert len(input_ids) == len(labels)
    assert len(input_ids) > 0
    
    # Find out exactly how many tokens the prompt should have taken
    prompt_tokens = tokenizer(record.prompt, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_tokens)
    
    # The prompt section should be entirely masked
    assert all(label == IGNORE_INDEX for label in labels[:prompt_len])
    
    # The completion section should NOT be masked
    assert all(label != IGNORE_INDEX for label in labels[prompt_len:])

def test_preprocess_record_truncation(tokenizer):
    """Test that the output strictly respects max_length."""
    record = PromptCompletion(
        prompt="A very long prompt that goes on and on " * 10,
        completion=" and finally ends."
    )
    
    max_length = 20
    result = preprocess_record(record, tokenizer, max_length=max_length)
    
    # Both input_ids and labels should be truncated to exactly max_length
    assert len(result["input_ids"]) == max_length
    assert len(result["labels"]) == max_length

def test_preprocess_record_prompt_exceeds_max_length(tokenizer):
    """Test the edge case where the prompt itself is longer than max_length."""
    record = PromptCompletion(
        prompt="This is a long prompt " * 10,
        completion=" Short completion"
    )

    max_length = 5
    result = preprocess_record(record, tokenizer, max_length=max_length)

    labels = result["labels"]

    # Because the prompt took up the entire max_length, ALL labels should be masked
    assert all(label == IGNORE_INDEX for label in labels)


@pytest.fixture(scope="module")
def special_token_tokenizer():
    """A tokenizer that wraps sequences in special tokens (BERT: [CLS] ... [SEP])."""
    return AutoTokenizer.from_pretrained("bert-base-uncased")


def test_preprocess_record_masks_correctly_with_special_tokens(special_token_tokenizer):
    """
    Regression: for tokenizers that prepend a special token (e.g. [CLS]/BOS),
    the prompt mask must include that leading special token so the boundary
    stays aligned and no completion token is wrongly masked / leaked.
    """
    tok = special_token_tokenizer
    record = PromptCompletion(
        prompt="Translate to French: Hello",
        completion=" Bonjour",
    )

    result = preprocess_record(record, tok, max_length=128)
    labels = result["labels"]

    # Leading special token(s) belong to the prompt span -> masked.
    assert labels[0] == IGNORE_INDEX

    # The boundary: prompt content tokens count, plus the leading special token.
    leading_special = 1  # BERT prepends [CLS]
    prompt_content_len = len(
        tok(record.prompt, add_special_tokens=False)["input_ids"]
    )
    boundary = leading_special + prompt_content_len

    assert all(label == IGNORE_INDEX for label in labels[:boundary])
    # First completion token must NOT be masked (off-by-one regression guard).
    assert labels[boundary] != IGNORE_INDEX
