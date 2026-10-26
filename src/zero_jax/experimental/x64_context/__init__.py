"""Frontend API routing for jax.experimental.x64_context."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import AbstractContextManager
from contextlib import contextmanager as _stdlib_contextmanager

from zero_jax._config_internal import enable_x64 as _enable_x64
from zero_jax._types import PyTree, VariadicCallable

from . import config


def contextmanager(func: VariadicCallable[Generator[PyTree, None, None]]) -> VariadicCallable[AbstractContextManager[PyTree]]:
    """Wrap a generator function as a context manager.

    Args:
        func: Generator function yielding a single value.

    Returns:
        VariadicCallable[AbstractContextManager[PyTree]]: Context manager factory.
    """
    return _stdlib_contextmanager(func)


def disable_x64() -> AbstractContextManager[None]:
    """Experimental context manager to temporarily disable X64 mode.

    Returns:
        AbstractContextManager[None]: Context manager disabling 64-bit precision.
    """
    return _enable_x64(False)


def enable_x64(new_val: bool = True) -> Generator[None, None, None]:
    """Experimental context manager to temporarily enable X64 mode.

    Args:
        new_val: Whether 64-bit mode should be enabled inside the context.

    Returns:
        Generator[None, None, None]: Context manager enabling 64-bit precision.
    """
    return _enable_x64(bool(new_val))


__all__ = ["config", "contextmanager", "disable_x64", "enable_x64"]
