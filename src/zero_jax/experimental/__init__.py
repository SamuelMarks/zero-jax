"""Frontend API routing for jax.experimental."""

from typing import Any


class EArray:
    """Mock implementation for EArray."""

    pass


from . import compilation_cache


def disable_x64(*args: Any, **kwargs: Any) -> Any:
    """Experimental context manager to temporarily disable X64 mode."""
    raise NotImplementedError("disable_x64 not yet implemented in zero-jax")


def enable_x64(*args: Any, **kwargs: Any) -> Any:
    """Experimental context manager to temporarily enable X64 mode."""
    raise NotImplementedError("enable_x64 not yet implemented in zero-jax")


def io_callback(*args: Any, **kwargs: Any) -> Any:
    """Calls an impure Python callback."""
    raise NotImplementedError("io_callback not yet implemented in zero-jax")


from . import x64_context
