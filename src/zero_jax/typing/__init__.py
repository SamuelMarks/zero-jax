"""Frontend API routing for jax.typing."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def ArrayLike(*args: Any, **kwargs: Any) -> Any:
    """Type annotation for JAX array-like objects."""
    return _ops.ArrayLike(*args, **kwargs)


def DTypeLike(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for DTypeLike.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.DTypeLike(*args, **kwargs)
