from .config import HatConfig
from .data import InvalidDatasetError, PromptCompletion, load_hf_dataset, load_jsonl

__all__ = [
    "HatConfig",
    "InvalidDatasetError",
    "PromptCompletion",
    "load_hf_dataset",
    "load_jsonl",
]

try:  # optional torch dependency
    from .model import HatEnabledModel  # noqa: F401

    __all__.append("HatEnabledModel")
except ModuleNotFoundError:
    pass
