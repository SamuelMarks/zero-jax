"""Frontend API routing for jax.debug."""

from typing import Any


class DebugEffect:
    """Mock implementation for DebugEffect."""

    pass


def breakpoint(*args: Any, **kwargs: Any) -> Any:
    """Enters a breakpoint at a point in a program."""
    raise NotImplementedError("breakpoint not yet implemented in zero-jax")


def callback(*args: Any, **kwargs: Any) -> Any:
    """Calls a stageable Python callback."""
    raise NotImplementedError("callback not yet implemented in zero-jax")


def inspect_array_sharding(*args: Any, **kwargs: Any) -> Any:
    """Enables inspecting array sharding inside JIT-ted functions."""
    raise NotImplementedError("inspect_array_sharding not yet implemented in zero-jax")


def print(*args: Any, **kwargs: Any) -> Any:
    """Prints values and works in staged out JAX functions."""
    raise NotImplementedError("print not yet implemented in zero-jax")


def visualize_array_sharding(*args: Any, **kwargs: Any) -> Any:
    """Visualizes an array's sharding."""
    raise NotImplementedError(
        "visualize_array_sharding not yet implemented in zero-jax"
    )


def visualize_sharding(*args: Any, **kwargs: Any) -> Any:
    """Visualizes a ``Sharding`` using ``rich``."""
    raise NotImplementedError("visualize_sharding not yet implemented in zero-jax")
