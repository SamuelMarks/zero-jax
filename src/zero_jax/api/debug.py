"""Debugging and leak checking API."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from zero_jax._config_internal import (
    config as _internal_config,
)
from zero_jax._config_internal import (
    numpy_dtype_promotion as _numpy_dtype_promotion,
)
from zero_jax._config_internal import (
    numpy_rank_promotion as _numpy_rank_promotion,
)
from zero_jax._config_internal import (
    transfer_guard as _transfer_guard,
)
from zero_jax.numpy.lax_numpy import ndarray


@contextlib.contextmanager
def check_tracer_leaks(enable: bool = True) -> Iterator[None]:
    """Context manager for checking tracer leaks.

    Args:
        enable: Whether to enable leak checking.

    Yields:
        None: Context where tracer leak checking is configured.
    """
    prev = bool(_internal_config.read("jax_check_tracer_leaks"))
    _internal_config.update("jax_check_tracer_leaks", enable)
    try:
        yield
    finally:
        _internal_config.update("jax_check_tracer_leaks", prev)


def checking_leaks() -> bool:
    """Return ``True`` if tracer leak checking is enabled.

    Returns:
        bool: Boolean indicating if leak checking is active.
    """
    return bool(_internal_config.read("jax_check_tracer_leaks"))


def clear_caches() -> None:
    """Clear all JAX compilation and staging caches."""


@contextlib.contextmanager
def debug_infs(enable: bool = True) -> Iterator[None]:
    """Context manager for debugging ``inf`` values.

    Args:
        enable: Whether to enable ``inf`` debugging.

    Yields:
        None: Context where ``inf`` debugging is configured.
    """
    prev = bool(_internal_config.read("jax_debug_infs"))
    _internal_config.update("jax_debug_infs", enable)
    try:
        yield
    finally:
        _internal_config.update("jax_debug_infs", prev)


@contextlib.contextmanager
def debug_nans(enable: bool = True) -> Iterator[None]:
    """Context manager for debugging ``NaN`` values.

    Args:
        enable: Whether to enable ``NaN`` debugging.

    Yields:
        None: Context where ``NaN`` debugging is configured.
    """
    prev = bool(_internal_config.read("jax_debug_nans"))
    _internal_config.update("jax_debug_nans", enable)
    try:
        yield
    finally:
        _internal_config.update("jax_debug_nans", prev)


def enable_checks(enable: bool = True) -> None:
    """Enable or disable JAX runtime invariant checks.

    Args:
        enable: Boolean indicating whether to enable checks.
    """
    _internal_config.update("jax_enable_checks", enable)


def print_environment_info() -> None:
    """Print information about the JAX environment and devices."""
    print("zero-jax environment: routing to ml-switcheroo-compiler")


def effects_barrier() -> None:
    """Block until all pending side effects have been executed."""


def live_arrays() -> list[ndarray]:
    """Return a list of all currently live JAX arrays.

    Returns:
        list[ndarray]: Empty list for ``zero-jax``.
    """
    return []


@contextlib.contextmanager
def log_compiles(enable: bool = True) -> Iterator[None]:
    """Context manager for logging compilation events.

    Args:
        enable: Whether to enable compilation logging.

    Yields:
        None: Context where compilation logging is configured.
    """
    prev = bool(_internal_config.read("jax_log_compiles"))
    _internal_config.update("jax_log_compiles", enable)
    try:
        yield
    finally:
        _internal_config.update("jax_log_compiles", prev)


@contextlib.contextmanager
def numpy_dtype_promotion(mode: str) -> Iterator[None]:
    """Context manager for setting NumPy dtype promotion mode.

    Args:
        mode: The promotion mode (``"standard"`` or ``"strict"``).

    Yields:
        None: Context where dtype promotion mode is set.
    """
    with _numpy_dtype_promotion(mode):
        yield


@contextlib.contextmanager
def numpy_rank_promotion(mode: str) -> Iterator[None]:
    """Context manager for setting NumPy rank promotion mode.

    Args:
        mode: The rank promotion mode (``"allow"``, ``"warn"``, or ``"raise"``).

    Yields:
        None: Context where rank promotion mode is set.
    """
    with _numpy_rank_promotion(mode):
        yield


@contextlib.contextmanager
def spmd_mode(mode: str) -> Iterator[None]:
    """Context manager for setting SPMD execution mode.

    Args:
        mode: The SPMD mode string.

    Yields:
        None: Context where SPMD mode is set.
    """
    prev = _internal_config.read("jax_spmd_mode")
    _internal_config.update("jax_spmd_mode", mode)
    try:
        yield
    finally:
        _internal_config.update("jax_spmd_mode", prev)


@contextlib.contextmanager
def transfer_guard(level: str) -> Iterator[None]:
    """Context manager for guarding against implicit device transfers.

    Args:
        level: The guard level (``"allow"``, ``"log"``, ``"disallow"``, etc.).

    Yields:
        None: Context where transfer guard level is active.
    """
    with _transfer_guard(level):
        yield


@contextlib.contextmanager
def transfer_guard_device_to_device(level: str) -> Iterator[None]:
    """Context manager guarding device-to-device transfers.

    Args:
        level: The guard level.

    Yields:
        None: Context where device-to-device transfer guard is active.
    """
    with _transfer_guard(level):
        yield


@contextlib.contextmanager
def transfer_guard_device_to_host(level: str) -> Iterator[None]:
    """Context manager guarding device-to-host transfers.

    Args:
        level: The guard level.

    Yields:
        None: Context where device-to-host transfer guard is active.
    """
    with _transfer_guard(level):
        yield


@contextlib.contextmanager
def transfer_guard_host_to_device(level: str) -> Iterator[None]:
    """Context manager guarding host-to-device transfers.

    Args:
        level: The guard level.

    Yields:
        None: Context where host-to-device transfer guard is active.
    """
    with _transfer_guard(level):
        yield


__all__ = [
    "check_tracer_leaks",
    "checking_leaks",
    "clear_caches",
    "debug_infs",
    "debug_nans",
    "effects_barrier",
    "enable_checks",
    "live_arrays",
    "log_compiles",
    "numpy_dtype_promotion",
    "numpy_rank_promotion",
    "print_environment_info",
    "spmd_mode",
    "transfer_guard",
    "transfer_guard_device_to_device",
    "transfer_guard_device_to_host",
    "transfer_guard_host_to_device",
]
