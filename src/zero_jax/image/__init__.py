"""Frontend API routing for jax.image."""

from typing import Any


class ResizeMethod:
    """Image resize method."""

    pass


def resize(*args: Any, **kwargs: Any) -> Any:
    """Image resize."""
    raise NotImplementedError("resize not yet implemented in zero-jax")


def scale_and_translate(*args: Any, **kwargs: Any) -> Any:
    """Apply a scale and translation to an image."""
    raise NotImplementedError("scale_and_translate not yet implemented in zero-jax")
