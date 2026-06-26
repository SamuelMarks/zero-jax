"""Frontend API routing for jax.experimental.x64_context."""

from typing import Any
from . import config


def contextmanager(*args: Any, **kwargs: Any) -> Any:
    """@contextmanager decorator."""
    raise NotImplementedError("contextmanager not yet implemented in zero-jax")


def disable_x64(*args: Any, **kwargs: Any) -> Any:
    """Experimental context manager to temporarily disable X64 mode."""
    raise NotImplementedError("disable_x64 not yet implemented in zero-jax")


def enable_x64(*args: Any, **kwargs: Any) -> Any:
    """Experimental context manager to temporarily enable X64 mode."""
    raise NotImplementedError("enable_x64 not yet implemented in zero-jax")
