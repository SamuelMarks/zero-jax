"""Frontend API routing for jax.debug."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


class DebugEffect:
    """Mock implementation for DebugEffect."""

    pass


def breakpoint(*args: Any, **kwargs: Any) -> Any:
    """Enters a breakpoint at a point in a program."""
    return getattr(_ops, "breakpoint")(*args, **kwargs)


def callback(*args: Any, **kwargs: Any) -> Any:
    """Calls a stageable Python callback."""
    return getattr(_ops, "callback")(*args, **kwargs)


def inspect_array_sharding(*args: Any, **kwargs: Any) -> Any:
    """Enables inspecting array sharding inside JIT-ted functions."""
    return getattr(_ops, "inspect_array_sharding")(*args, **kwargs)


def print(*args: Any, **kwargs: Any) -> Any:
    """Prints values and works in staged out JAX functions."""
    return getattr(_ops, "print")(*args, **kwargs)


def visualize_array_sharding(*args: Any, **kwargs: Any) -> Any:
    """Visualizes an array's sharding."""
    return getattr(_ops, "visualize_array_sharding")(*args, **kwargs)


def visualize_sharding(*args: Any, **kwargs: Any) -> Any:
    """Visualizes a ``Sharding`` using ``rich``."""
    return getattr(_ops, "visualize_sharding")(*args, **kwargs)
