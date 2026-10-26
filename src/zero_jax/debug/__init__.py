"""Frontend API routing for jax.debug."""

from __future__ import annotations

import builtins

from zero_jax._types import ArrayLikeSpec, ConfigValue, VariadicCallable


class DebugEffect:
    """Frontend state holder for DebugEffect."""

    def __init__(
        self,
        *,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize ``DebugEffect`` instance.

        Args:
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
        """
        self.id = id
        self.name = name
        self.value = value


def breakpoint(*, backend: str | None = None, filter_frames: bool = True, **kwargs: ArrayLikeSpec) -> None:
    """Enter a breakpoint at a point in a program.

    Args:
        backend: Optional debugger backend name.
        filter_frames: Whether to filter internal JAX frames.
        **kwargs: Additional keyword arguments.
    """
    _ = (backend, filter_frames, kwargs)


def callback(
    callback_fn: VariadicCallable[ArrayLikeSpec],
    *args: ArrayLikeSpec,
    ordered: bool = False,
    partitioned: bool = False,
    **kwargs: ArrayLikeSpec,
) -> None:
    """Call a stageable Python callback for side effects.

    Args:
        callback_fn: Python callable to invoke.
        *args: Positional arguments forwarded to ``callback_fn``.
        ordered: Whether callback execution order is preserved.
        partitioned: Whether callback operates on local shards.
        **kwargs: Keyword arguments forwarded to ``callback_fn``.
    """
    _ = (ordered, partitioned)
    callback_fn(*args, **kwargs)


def inspect_array_sharding(value: ArrayLikeSpec, *, callback: VariadicCallable[ArrayLikeSpec]) -> None:
    """Enable inspecting array sharding inside JIT-ted functions.

    Args:
        value: Array whose sharding is inspected.
        callback: Function receiving the sharding object.
    """
    _ = value
    callback(None)


def print(fmt: str, *args: ArrayLikeSpec, ordered: bool = False, **kwargs: ArrayLikeSpec) -> None:
    """Print formatted values inside staged JAX functions.

    Args:
        fmt: Format string.
        *args: Positional format arguments.
        ordered: Whether print order is preserved across effects.
        **kwargs: Keyword format arguments.
    """
    _ = ordered
    builtins.print(fmt.format(*args, **kwargs))


def visualize_array_sharding(arr: ArrayLikeSpec, *, use_color: bool = True) -> str:
    """Visualize an array's sharding as a string.

    Args:
        arr: Array to visualize.
        use_color: Whether to include ANSI color codes.

    Returns:
        str: String representation of ``arr`` sharding.
    """
    _ = use_color
    return str(arr)


def visualize_sharding(shape: ConfigValue, sharding: ConfigValue = None, *, use_color: bool = True) -> str:
    """Visualize a ``Sharding`` for a given ``shape``.

    Args:
        shape: Shape tuple or sharding object.
        sharding: Optional sharding specification.
        use_color: Whether to include ANSI color codes.

    Returns:
        str: String representation of the sharding.
    """
    _ = (sharding, use_color)
    return str(shape)


__all__ = [
    "DebugEffect",
    "breakpoint",
    "callback",
    "inspect_array_sharding",
    "print",
    "visualize_array_sharding",
    "visualize_sharding",
]
