"""Frontend API routing for jax.experimental.x64_context.config.logging_config."""

from typing import Any


def disable_all_debug_logging(*args: Any, **kwargs: Any) -> Any:
    """Disables all debug logging enabled via `enable_debug_logging`."""
    raise NotImplementedError(
        "disable_all_debug_logging not yet implemented in zero-jax"
    )


def enable_debug_logging(*args: Any, **kwargs: Any) -> Any:
    """Makes the specified logger log everything to stderr."""
    raise NotImplementedError("enable_debug_logging not yet implemented in zero-jax")


from . import logging
from . import sys
