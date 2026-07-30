"""Frontend API routing for jax.experimental.x64_context.config.logging_config."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def disable_all_debug_logging(*args: Any, **kwargs: Any) -> Any:
    """Disables all debug logging enabled via `enable_debug_logging`."""
    return _ops.disable_all_debug_logging(*args, **kwargs)  # pragma: no cover


def enable_debug_logging(*args: Any, **kwargs: Any) -> Any:
    """Makes the specified logger log everything to stderr."""
    return _ops.enable_debug_logging(*args, **kwargs)  # pragma: no cover


from . import logging, sys
