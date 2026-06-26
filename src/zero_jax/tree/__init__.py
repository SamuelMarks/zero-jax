"""Mock implementation for jax.tree."""

from typing import Any


def all(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for all."""
    raise NotImplementedError("all not yet implemented in zero-jax")


def flatten(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for flatten."""
    raise NotImplementedError("flatten not yet implemented in zero-jax")


def leaves(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for leaves."""
    raise NotImplementedError("leaves not yet implemented in zero-jax")


def map(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for map."""
    raise NotImplementedError("map not yet implemented in zero-jax")


def reduce(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for reduce."""
    raise NotImplementedError("reduce not yet implemented in zero-jax")


def structure(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for structure."""
    raise NotImplementedError("structure not yet implemented in zero-jax")


def transpose(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for transpose."""
    raise NotImplementedError("transpose not yet implemented in zero-jax")


def unflatten(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for unflatten."""
    raise NotImplementedError("unflatten not yet implemented in zero-jax")


__all__ = [
    "all",
    "flatten",
    "leaves",
    "map",
    "reduce",
    "structure",
    "transpose",
    "unflatten",
]
