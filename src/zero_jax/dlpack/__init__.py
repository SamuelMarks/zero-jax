"""Frontend API routing for jax.dlpack."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


class SUPPORTED_DTYPES:
    """Supported dlpack dtypes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


def from_dlpack(*args: Any, **kwargs: Any) -> Any:
    """Returns a :class:`~jax.Array` representation of a DLPack tensor."""
    return _ops.from_dlpack(*args, **kwargs)


def to_dlpack(*args: Any, **kwargs: Any) -> Any:
    """Returns a DLPack tensor that encapsulates a :class:`~jax.Array` ``x``."""
    return _ops.to_dlpack(*args, **kwargs)
