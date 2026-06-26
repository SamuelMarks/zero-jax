"""Mock implementation for jax.lib.xla_bridge."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops

from . import xla_client


def default_backend(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for default_backend."""
    return getattr(_ops, "default_backend")(*args, **kwargs)


def get_backend(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for get_backend."""
    return getattr(_ops, "get_backend")(*args, **kwargs)


def get_compile_options(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for get_compile_options."""
    return getattr(_ops, "get_compile_options")(*args, **kwargs)


__all__ = ["default_backend", "get_backend", "get_compile_options", "xla_client"]
