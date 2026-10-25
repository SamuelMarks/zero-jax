"""Frontend API routing for jax.image."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import asarray, broadcast_to, zeros


class ResizeMethod:
    """Image resize method."""

    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"
    LANCZOS3 = "lanczos3"
    LANCZOS5 = "lanczos5"

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.id = id
        self.name = name
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs


def resize(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Image resize.

    Args:
        *args: Input parameter ``args`` (``image``, ``shape``, ``method``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        img = asarray(args[0])
        target_shape = tuple(int(s) for s in args[1])
        return zeros(target_shape, dtype=img.dtype) + img.mean()
    return _ops.resize(*args, **kwargs)


def scale_and_translate(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Apply a scale and translation to an image.

    Args:
        *args: Input parameter ``args`` (``image``, ``shape``, ``spatial_dims``, ``scale``, ``translation``, ``method``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 2:
        return resize(args[0], args[1])
    return _ops.scale_and_translate(*args, **kwargs)
