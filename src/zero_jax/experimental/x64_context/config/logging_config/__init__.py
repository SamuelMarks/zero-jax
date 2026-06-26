"""Frontend API routing for jax.experimental.x64_context.config.logging_config."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def disable_all_debug_logging(*args: Any, **kwargs: Any) -> Any:
    """Disables all debug logging enabled via `enable_debug_logging`."""
    return getattr(_ops, "disable_all_debug_logging")(*args, **kwargs)


def enable_debug_logging(*args: Any, **kwargs: Any) -> Any:
    """Makes the specified logger log everything to stderr."""
    return getattr(_ops, "enable_debug_logging")(*args, **kwargs)


from . import logging
from . import sys
