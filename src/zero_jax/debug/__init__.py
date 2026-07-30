"""Frontend API routing for jax.debug."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


class DebugEffect:
    """Frontend state holder for DebugEffect."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


def breakpoint(*args: Any, **kwargs: Any) -> Any:
    """Enters a breakpoint at a point in a program."""
    return _ops.breakpoint(*args, **kwargs)


def callback(*args: Any, **kwargs: Any) -> Any:
    """Calls a stageable Python callback."""
    return _ops.callback(*args, **kwargs)


def inspect_array_sharding(*args: Any, **kwargs: Any) -> Any:
    """Enables inspecting array sharding inside JIT-ted functions."""
    return _ops.inspect_array_sharding(*args, **kwargs)


def print(*args: Any, **kwargs: Any) -> Any:
    """Prints values and works in staged out JAX functions."""
    return _ops.print(*args, **kwargs)


def visualize_array_sharding(*args: Any, **kwargs: Any) -> Any:
    """Visualizes an array's sharding."""
    return _ops.visualize_array_sharding(*args, **kwargs)


def visualize_sharding(*args: Any, **kwargs: Any) -> Any:
    """Visualizes a ``Sharding`` using ``rich``."""
    return _ops.visualize_sharding(*args, **kwargs)
