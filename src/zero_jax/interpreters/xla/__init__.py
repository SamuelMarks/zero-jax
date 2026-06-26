"""Frontend API routing for jax.interpreters.xla."""

from typing import Any


class Backend:
    """Mock implementation for Backend."""

    pass


def abstractify(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for abstractify."""
    raise NotImplementedError("abstractify not yet implemented in zero-jax")


def apply_primitive(*args: Any, **kwargs: Any) -> Any:
    """Impl rule that compiles and runs a single primitive 'prim' using XLA."""
    raise NotImplementedError("apply_primitive not yet implemented in zero-jax")


def canonicalize_dtype(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for canonicalize_dtype."""
    raise NotImplementedError("canonicalize_dtype not yet implemented in zero-jax")


canonicalize_dtype_handlers: Any = None

pytype_aval_mappings: Any = None
from . import xb
from . import xc
from . import xe
