"""Frontend API routing for jax.experimental."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


class EArray:
    """Mock implementation for EArray."""

    pass


from . import compilation_cache


def disable_x64(*args: Any, **kwargs: Any) -> Any:
    """Experimental context manager to temporarily disable X64 mode."""
    return getattr(_ops, "disable_x64")(*args, **kwargs)


def enable_x64(*args: Any, **kwargs: Any) -> Any:
    """Experimental context manager to temporarily enable X64 mode."""
    return getattr(_ops, "enable_x64")(*args, **kwargs)


def io_callback(*args: Any, **kwargs: Any) -> Any:
    """Calls an impure Python callback."""
    return getattr(_ops, "io_callback")(*args, **kwargs)


from . import x64_context
