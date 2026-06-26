"""Frontend API routing for jax.image."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


class ResizeMethod:
    """Image resize method."""

    pass


def resize(*args: Any, **kwargs: Any) -> Any:
    """Image resize."""
    return getattr(_ops, "resize")(*args, **kwargs)


def scale_and_translate(*args: Any, **kwargs: Any) -> Any:
    """Apply a scale and translation to an image."""
    return getattr(_ops, "scale_and_translate")(*args, **kwargs)
