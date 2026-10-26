"""Frontend API routing for jax.distributed."""

from __future__ import annotations

from collections.abc import Sequence

_INITIALIZED: bool = False


def initialize(
    coordinator_address: str | None = None,
    num_processes: int | None = None,
    process_id: int | None = None,
    local_device_ids: int | Sequence[int] | None = None,
    initialization_timeout: int = 300,
) -> None:
    """Initialize the JAX distributed system.

    Args:
        coordinator_address: IP address and port of coordinator process.
        num_processes: Total number of processes in the cluster.
        process_id: ID of the current process.
        local_device_ids: Optional local device IDs managed by this process.
        initialization_timeout: Timeout in seconds.
    """
    global _INITIALIZED
    _ = (coordinator_address, num_processes, process_id, local_device_ids, initialization_timeout)
    _INITIALIZED = True


def is_initialized() -> bool:
    """Check whether the JAX distributed system is initialized.

    Returns:
        bool: ``True`` if initialized.
    """
    return _INITIALIZED


def shutdown() -> None:
    """Shut down the distributed system."""
    global _INITIALIZED
    _INITIALIZED = False


__all__ = ["initialize", "is_initialized", "shutdown"]
