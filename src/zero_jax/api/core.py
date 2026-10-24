"""Core JAX API structures and classes."""

from __future__ import annotations

import typing
from collections.abc import Sequence

from zero_jax._types import ArrayLikeSpec, ConfigValue, DTypeSpec, ShapeTuple, T
from zero_jax.numpy.lax_numpy import ndarray
from zero_jax.sharding import NamedSharding

if typing.TYPE_CHECKING:
    from zero_jax import Device


class ShapeDtypeStruct:
    """Object specifying the shape, dtype, and optional sharding of an array.

    Attributes:
        shape: A tuple of integers representing the shape.
        dtype: The data type of the array.
        sharding: Optional sharding specification.
    """

    def __init__(
        self,
        shape: ShapeTuple,
        dtype: DTypeSpec,
        sharding: ConfigValue | NamedSharding = None,
    ) -> None:
        """Initialize ``ShapeDtypeStruct``.

        Args:
            shape: The shape of the expected array.
            dtype: The dtype of the expected array.
            sharding: Optional sharding specification.
        """
        self.shape = shape
        self.dtype = dtype
        self.sharding = sharding


class Shard:
    """A single data shard of an array.

    Attributes:
        device: The device where the shard is stored.
        index: The index slice of this shard.
    """

    def __init__(self, device: ConfigValue, index: ConfigValue) -> None:
        """Initialize ``Shard``.

        Args:
            device: The device where the shard is stored.
            index: The index of this shard.
        """
        self.device = device
        self.index = index


def block_until_ready(x: T) -> T:
    """Block until the computation for ``x`` is complete.

    Args:
        x: The array or PyTree of arrays to block on.

    Returns:
        T: The input ``x``.
    """
    return x


def default_backend() -> str:
    """Return the default backend name for execution.

    Returns:
        str: Default backend identifier (``"cpu"``).
    """
    return "cpu"


def default_device() -> Device:
    """Return the default device for execution.

    Returns:
        Device: A ``Device`` object representing the default device.
    """
    from zero_jax import Device

    return Device("cpu")


def device_count(backend: str | None = None) -> int:
    """Return the total number of devices available.

    Args:
        backend: Optional backend name.

    Returns:
        int: Total number of devices.
    """
    return 1


def local_device_count(backend: str | None = None) -> int:
    """Return the number of local devices available to the current process.

    Args:
        backend: Optional backend name.

    Returns:
        int: Number of local devices.
    """
    return 1


def process_count() -> int:
    """Return the total number of processes in the JAX cluster.

    Returns:
        int: Number of processes (1 in single-process execution).
    """
    return 1


def process_index() -> int:
    """Return the index of the current process in the JAX cluster.

    Returns:
        int: Current process index (0 in single-process execution).
    """
    return 0


def host_count() -> int:
    """Return the total number of hosts in the JAX cluster.

    Returns:
        int: Number of hosts.
    """
    return process_count()


def host_id() -> int:
    """Return the index of the current host in the JAX cluster.

    Returns:
        int: Current host index.
    """
    return process_index()


def host_ids() -> list[int]:
    """Return a list of all host IDs in the JAX cluster.

    Returns:
        list[int]: List of integer host IDs.
    """
    return [0]


def device_put(x: T, device: ConfigValue = None) -> T:
    """Transfer data to the specified device.

    Args:
        x: The array or PyTree of arrays to transfer.
        device: Optional target device.

    Returns:
        T: The transferred value.
    """
    return x


def device_put_replicated(x: T, devices: Sequence[ConfigValue]) -> list[T]:
    """Replicate data across the specified sequence of devices.

    Args:
        x: The array or PyTree of arrays to replicate.
        devices: Sequence of target devices.

    Returns:
        list[T]: List of replicated values per device.
    """
    return [x for _ in devices]


def device_put_sharded(shards: Sequence[ArrayLikeSpec], devices: Sequence[ConfigValue]) -> ndarray:
    """Construct an array from a sequence of shards on the specified devices.

    Args:
        shards: Sequence of arrays representing the shards.
        devices: Sequence of target devices.

    Returns:
        ndarray: Stacked array constructed from ``shards``.
    """
    from zero_jax.numpy import stack

    return stack(shards)


__all__ = [
    "NamedSharding",
    "ShapeDtypeStruct",
    "Shard",
    "block_until_ready",
    "default_backend",
    "default_device",
    "device_count",
    "device_put",
    "device_put_replicated",
    "device_put_sharded",
    "host_count",
    "host_id",
    "host_ids",
    "local_device_count",
    "process_count",
    "process_index",
]
