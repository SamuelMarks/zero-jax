"""Frontend API routing for jax.dlpack."""

from typing import Any

SUPPORTED_DTYPES: Any = None


def from_dlpack(*args: Any, **kwargs: Any) -> Any:
    """Returns a :class:`~jax.Array` representation of a DLPack tensor."""
    raise NotImplementedError("from_dlpack not yet implemented in zero-jax")


def to_dlpack(*args: Any, **kwargs: Any) -> Any:
    """Returns a DLPack tensor that encapsulates a :class:`~jax.Array` ``x``."""
    raise NotImplementedError("to_dlpack not yet implemented in zero-jax")
