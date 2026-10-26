"""Frontend API routing for jax.lib.xla_bridge."""

from __future__ import annotations

from zero_jax._types import ConfigValue
from zero_jax.api.core import default_backend as _core_default_backend
from zero_jax.lax import Device

from . import xla_client


def default_backend() -> str:
    """Return the platform name of the default XLA backend.

    Returns:
        str: Default backend platform string (e.g. ``"cpu"``).
    """
    return _core_default_backend()


def get_backend(platform: str | None = None) -> Device:
    """Return the XLA backend client/device for ``platform``.

    Args:
        platform: Optional backend platform name (defaults to ``"cpu"``).

    Returns:
        Device: Backend device descriptor.
    """
    return Device(platform=platform or "cpu")


def get_compile_options(
    num_replicas: int = 1,
    num_partitions: int = 1,
    device_assignment: tuple[int, ...] | None = None,
    use_spmd_partitioning: bool = True,
) -> dict[str, ConfigValue]:
    """Return a dictionary of XLA compilation options.

    Args:
        num_replicas: Number of computation replicas.
        num_partitions: Number of model partitions.
        device_assignment: Optional tuple of device IDs.
        use_spmd_partitioning: Whether SPMD partitioning is enabled.

    Returns:
        dict[str, ConfigValue]: Compilation options mapping.
    """
    return {
        "num_replicas": num_replicas,
        "num_partitions": num_partitions,
        "device_assignment": device_assignment,
        "use_spmd_partitioning": use_spmd_partitioning,
    }


__all__ = ["default_backend", "get_backend", "get_compile_options", "xla_client"]
