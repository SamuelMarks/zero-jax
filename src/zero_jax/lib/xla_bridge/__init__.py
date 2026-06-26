"""Mock implementation for jax.lib.xla_bridge."""

from typing import Any

from . import xla_client


def default_backend(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for default_backend."""
    raise NotImplementedError("default_backend not yet implemented in zero-jax")


def get_backend(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for get_backend."""
    raise NotImplementedError("get_backend not yet implemented in zero-jax")


def get_compile_options(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for get_compile_options."""
    raise NotImplementedError("get_compile_options not yet implemented in zero-jax")


__all__ = ["default_backend", "get_backend", "get_compile_options", "xla_client"]
