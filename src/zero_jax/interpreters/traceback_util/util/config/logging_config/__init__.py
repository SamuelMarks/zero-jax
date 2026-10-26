"""Frontend API routing for jax.interpreters.traceback_util.util.config.logging_config."""

from __future__ import annotations

from zero_jax._config_internal import disable_all_debug_logging, enable_debug_logging

__all__ = [
    "disable_all_debug_logging",
    "enable_debug_logging",
]
