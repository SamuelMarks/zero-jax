"""Frontend API routing for jax.dlpack."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue


class SUPPORTED_DTYPES:
    """Supported dlpack dtypes."""

    def __init__(self, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> None:
        """Execute ``__init__``.

        Args:
            *args: Input parameter ``args``.
            **kwargs: Input parameter ``kwargs``.
        """


def from_dlpack(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Returns a :class:`~jax.Array` representation of a DLPack tensor.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.from_dlpack(*args, **kwargs)


def to_dlpack(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Returns a DLPack tensor that encapsulates a :class:`~jax.Array` ``x``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.to_dlpack(*args, **kwargs)
