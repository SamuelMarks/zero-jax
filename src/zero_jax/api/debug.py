"""Debugging and leak checking API."""

from __future__ import annotations

from typing import Any, Callable
import contextlib


@contextlib.contextmanager
def check_tracer_leaks(enable: bool = True) -> Any:
    """Context manager for checking tracer leaks.

    Args:
        enable: Whether to enable leak checking.

    Yields:
        None
    """
    yield


def checking_leaks() -> bool:
    """Returns True if tracer leak checking is enabled.

    Returns:
        Boolean indicating if leak checking is active.
    """
    return False


def clear_caches() -> None:
    """Clears all JAX compilation and staging caches."""
    pass


@contextlib.contextmanager
def debug_infs(enable: bool = True) -> Any:
    """Context manager for debugging INFs.

    Args:
        enable: Whether to enable INF debugging.

    Yields:
        None
    """
    yield


@contextlib.contextmanager
def debug_nans(enable: bool = True) -> Any:
    """Context manager for debugging NANs.

    Args:
        enable: Whether to enable NAN debugging.

    Yields:
        None
    """
    yield


def enable_checks(enable: bool = True) -> None:
    """Enables or disables various JAX runtime checks.

    Args:
        enable: Boolean indicating whether to enable checks.
    """
    pass


def print_environment_info() -> None:
    """Prints information about the JAX environment and devices."""
    print("zero-jax environment: routing to ml-switcheroo-compiler")


def effects_barrier() -> None:
    """Blocks until all effects have been executed."""
    pass


def live_arrays() -> list[Any]:
    """Returns a list of all currently live JAX arrays.

    Returns:
        An empty list for zero-jax.
    """
    return []


@contextlib.contextmanager
def log_compiles(enable: bool = True) -> Any:
    """Context manager for logging compilation events.

    Args:
        enable: Whether to enable compilation logging.

    Yields:
        None
    """
    yield


@contextlib.contextmanager
def numpy_dtype_promotion(mode: str) -> Any:
    """Context manager for setting numpy dtype promotion mode.

    Args:
        mode: The promotion mode ('standard' or 'strict').

    Yields:
        None
    """
    yield


@contextlib.contextmanager
def numpy_rank_promotion(mode: str) -> Any:
    """Context manager for setting numpy rank promotion mode.

    Args:
        mode: The promotion mode.

    Yields:
        None
    """
    yield


@contextlib.contextmanager
def spmd_mode(mode: str) -> Any:
    """Context manager for setting SPMD mode.

    Args:
        mode: The SPMD mode string.

    Yields:
        None
    """
    yield


@contextlib.contextmanager
def transfer_guard(level: str) -> Any:
    """Context manager for guarding against implicit device transfers.

    Args:
        level: The guard level ('allow', 'warn', 'disallow', etc.).

    Yields:
        None
    """
    yield


@contextlib.contextmanager
def transfer_guard_device_to_device(level: str) -> Any:
    """Guard for device-to-device transfers.

    Args:
        level: Guard level.

    Yields:
        None
    """
    yield


@contextlib.contextmanager
def transfer_guard_device_to_host(level: str) -> Any:
    """Guard for device-to-host transfers.

    Args:
        level: Guard level.

    Yields:
        None
    """
    yield


@contextlib.contextmanager
def transfer_guard_host_to_device(level: str) -> Any:
    """Guard for host-to-device transfers.

    Args:
        level: Guard level.

    Yields:
        None
    """
    yield
