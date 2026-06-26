"""Frontend API routing for jax.tree."""

from typing import Any


def all(*args: Any, **kwargs: Any) -> Any:
    """Call all() over the leaves of a tree."""
    raise NotImplementedError("all not yet implemented in zero-jax")


def flatten(*args: Any, **kwargs: Any) -> Any:
    """Flattens a pytree."""
    raise NotImplementedError("flatten not yet implemented in zero-jax")


def leaves(*args: Any, **kwargs: Any) -> Any:
    """Gets the leaves of a pytree."""
    raise NotImplementedError("leaves not yet implemented in zero-jax")


def map(*args: Any, **kwargs: Any) -> Any:
    """Maps a multi-input function over pytree args to produce a new pytree."""
    raise NotImplementedError("map not yet implemented in zero-jax")


def reduce(*args: Any, **kwargs: Any) -> Any:
    """Call reduce() over the leaves of a tree."""
    raise NotImplementedError("reduce not yet implemented in zero-jax")


def structure(*args: Any, **kwargs: Any) -> Any:
    """Gets the treedef for a pytree."""
    raise NotImplementedError("structure not yet implemented in zero-jax")


def transpose(*args: Any, **kwargs: Any) -> Any:
    """Transform a tree having tree structure (outer, inner) into one having structure (inner, outer)."""
    raise NotImplementedError("transpose not yet implemented in zero-jax")


def unflatten(*args: Any, **kwargs: Any) -> Any:
    """Reconstructs a pytree from the treedef and the leaves."""
    raise NotImplementedError("unflatten not yet implemented in zero-jax")
