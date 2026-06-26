"""Frontend API routing for jax.experimental.x64_context."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops
from . import config


def contextmanager(*args: Any, **kwargs: Any) -> Any:
    """@contextmanager decorator."""
    return getattr(_ops, "contextmanager")(*args, **kwargs)


def disable_x64(*args: Any, **kwargs: Any) -> Any:
    """Experimental context manager to temporarily disable X64 mode."""
    return getattr(_ops, "disable_x64")(*args, **kwargs)


def enable_x64(*args: Any, **kwargs: Any) -> Any:
    """Experimental context manager to temporarily enable X64 mode."""
    return getattr(_ops, "enable_x64")(*args, **kwargs)
