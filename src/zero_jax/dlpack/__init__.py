"""Frontend API routing for jax.dlpack."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops

SUPPORTED_DTYPES: Any = None


def from_dlpack(*args: Any, **kwargs: Any) -> Any:
    """Returns a :class:`~jax.Array` representation of a DLPack tensor."""
    return getattr(_ops, "from_dlpack")(*args, **kwargs)


def to_dlpack(*args: Any, **kwargs: Any) -> Any:
    """Returns a DLPack tensor that encapsulates a :class:`~jax.Array` ``x``."""
    return getattr(_ops, "to_dlpack")(*args, **kwargs)
