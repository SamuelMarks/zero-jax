"""Frontend API routing for jax.experimental.x64_context.config.logging_config."""

from __future__ import annotations

from zero_jax._config_internal import disable_all_debug_logging, enable_debug_logging

from . import logging, sys

__all__ = [
    "disable_all_debug_logging",
    "enable_debug_logging",
    "logging",
    "sys",
]
