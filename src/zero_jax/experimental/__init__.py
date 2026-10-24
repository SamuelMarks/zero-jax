"""Frontend API routing for jax.experimental."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue


class EArray:
    """Frontend state holder for EArray."""

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


from . import compilation_cache


def disable_x64(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Experimental context manager to temporarily disable X64 mode.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.disable_x64(*args, **kwargs)


def enable_x64(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Experimental context manager to temporarily enable X64 mode.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.enable_x64(*args, **kwargs)


def io_callback(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Calls an impure Python callback.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.io_callback(*args, **kwargs)


from . import x64_context


class BlurConfig:
    """Frontend class for ``BlurConfig``."""


class CTCLossOptions:
    """Frontend class for ``CTCLossOptions``."""


class ElasticConfig:
    """Frontend class for ``ElasticConfig``."""


class PerspectiveConfig:
    """Frontend class for ``PerspectiveConfig``."""
