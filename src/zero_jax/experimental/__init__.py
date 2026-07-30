"""Frontend API routing for jax.experimental."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


class EArray:
    """Frontend state holder for EArray."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():  # pragma: no cover
            setattr(self, k, v)  # pragma: no cover


from . import compilation_cache


def disable_x64(*args: Any, **kwargs: Any) -> Any:
    """Experimental context manager to temporarily disable X64 mode."""
    return _ops.disable_x64(*args, **kwargs)  # pragma: no cover


def enable_x64(*args: Any, **kwargs: Any) -> Any:
    """Experimental context manager to temporarily enable X64 mode."""
    return _ops.enable_x64(*args, **kwargs)  # pragma: no cover


def io_callback(*args: Any, **kwargs: Any) -> Any:
    """Calls an impure Python callback."""
    return _ops.io_callback(*args, **kwargs)  # pragma: no cover


from . import x64_context


class BlurConfig:
    pass


class CTCLossOptions:
    pass


class ElasticConfig:
    pass


class PerspectiveConfig:
    pass
