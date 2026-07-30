"""Frontend API routing for jax.experimental.x64_context."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops

from . import config


def contextmanager(*args: Any, **kwargs: Any) -> Any:
    """@contextmanager decorator."""
    return _ops.contextmanager(*args, **kwargs)  # pragma: no cover


def disable_x64(*args: Any, **kwargs: Any) -> Any:
    """Experimental context manager to temporarily disable X64 mode."""
    return _ops.disable_x64(*args, **kwargs)  # pragma: no cover


def enable_x64(*args: Any, **kwargs: Any) -> Any:
    """Experimental context manager to temporarily enable X64 mode."""
    return _ops.enable_x64(*args, **kwargs)  # pragma: no cover
