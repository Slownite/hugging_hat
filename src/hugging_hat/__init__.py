from .config import HatConfig

__all__ = ["HatConfig"]

try:  # optional torch dependency
    from .model import HatEnabledModel  # noqa: F401

    __all__.append("HatEnabledModel")
except ModuleNotFoundError:
    pass
