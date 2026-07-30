"""Frontend API routing for jax.custom_batching."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


class custom_vmap:
    """Frontend state holder for custom_vmap."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


def sequential_vmap(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for sequential_vmap.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Any: Result.
    """
    return _ops.sequential_vmap(*args, **kwargs)
