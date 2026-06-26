"""Frontend API routing for jax.interpreters.xla."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


class Backend:
    """Mock implementation for Backend."""

    pass


def abstractify(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for abstractify."""
    return getattr(_ops, "abstractify")(*args, **kwargs)


def apply_primitive(*args: Any, **kwargs: Any) -> Any:
    """Impl rule that compiles and runs a single primitive 'prim' using XLA."""
    return getattr(_ops, "apply_primitive")(*args, **kwargs)


def canonicalize_dtype(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for canonicalize_dtype."""
    return getattr(_ops, "canonicalize_dtype")(*args, **kwargs)


canonicalize_dtype_handlers: Any = None

pytype_aval_mappings: Any = None
from . import xb
from . import xc
from . import xe
