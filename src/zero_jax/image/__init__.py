"""Frontend API routing for jax.image."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


class ResizeMethod:
    """Image resize method."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


def resize(*args: Any, **kwargs: Any) -> Any:
    """Image resize."""
    return _ops.resize(*args, **kwargs)


def scale_and_translate(*args: Any, **kwargs: Any) -> Any:
    """Apply a scale and translation to an image."""
    return _ops.scale_and_translate(*args, **kwargs)
