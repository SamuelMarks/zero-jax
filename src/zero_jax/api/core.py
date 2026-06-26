"""Core JAX API structures and classes."""

from __future__ import annotations

from typing import Any, Tuple


class ShapeDtypeStruct:
    """An object specifying the shape and dtype of an array.

    Attributes:
        shape: A tuple of integers representing the shape.
        dtype: The data type of the array.
    """

    def __init__(
        self, shape: Tuple[int, ...], dtype: Any, sharding: Any = None
    ) -> None:
        """Initializes the ShapeDtypeStruct.

        Args:
            shape: The shape of the expected array.
            dtype: The dtype of the expected array.
            sharding: Optional sharding specification.
        """
        self.shape = shape
        self.dtype = dtype
        self.sharding = sharding


class Shard:
    """A single data shard of an array."""

    def __init__(self, device: Any, index: Any) -> None:
        """Initializes a Shard.

        Args:
            device: The device where the shard is stored.
            index: The index of this shard.
        """
        self.device = device
        self.index = index


class NamedSharding:
    """A sharding that specifies how an array is partitioned across devices."""

    def __init__(self, mesh: Any, spec: Any) -> None:
        """Initializes NamedSharding.

        Args:
            mesh: The device mesh.
            spec: The partition specification.
        """
        self.mesh = mesh
        self.spec = spec


def block_until_ready(x: Any) -> Any:
    """Blocks until the computation for x is complete.

    Since zero-jax executes eagerly or blocks by default on evaluation,
    this simply returns the input.

    Args:
        x: The array or tree of arrays to block on.

    Returns:
        The input x.
    """
    return x


def default_backend() -> str:
    """Returns the default backend for execution.

    Returns:
        A string representing the default backend ('cpu' by default).
    """
    return "cpu"


def default_device() -> Any:
    """Returns the default device for execution.

    Returns:
        A Device object representing the default device.
    """
    from zero_jax import Device  # pragma: no cover

    return Device("cpu")  # pragma: no cover


def device_count(backend: Any = None) -> int:
    """Returns the number of devices available.

    Args:
        backend: Optional backend name.

    Returns:
        The total number of devices.
    """
    return 1


def local_device_count(backend: Any = None) -> int:
    """Returns the number of local devices available.

    Args:
        backend: Optional backend name.

    Returns:
        The total number of local devices.
    """
    return 1


def process_count() -> int:
    """Returns the total number of processes in the JAX cluster.

    Returns:
        The number of processes (1 in non-distributed).
    """
    return 1


def process_index() -> int:
    """Returns the index of the current process in the JAX cluster.

    Returns:
        The index of the current process (0 in non-distributed).
    """
    return 0


def host_count() -> int:
    """Returns the total number of hosts in the JAX cluster.

    Returns:
        The number of hosts (1 in non-distributed).
    """
    return process_count()


def host_id() -> int:
    """Returns the index of the current host in the JAX cluster.

    Returns:
        The index of the current host (0 in non-distributed).
    """
    return process_index()


def host_ids() -> list[int]:
    """Returns a list of all host IDs in the JAX cluster.

    Returns:
        A list of integer host IDs.
    """
    return [0]


def device_put(x: Any, device: Any = None) -> Any:
    """Transfers data to the specified device.

    Args:
        x: The array or tree of arrays to transfer.
        device: The target device. If None, the default device is used.

    Returns:
        The array(s) on the specified device.
    """
    return x


def device_put_replicated(x: Any, devices: Any) -> Any:
    """Replicates data across the specified devices.

    Args:
        x: The array or tree of arrays to replicate.
        devices: A sequence of target devices.

    Returns:
        A sequence of arrays replicated on the target devices.
    """
    return [x for _ in devices]


def device_put_sharded(shards: Any, devices: Any) -> Any:
    """Constructs an array from a sequence of shards on the specified devices.

    Args:
        shards: A sequence of arrays representing the shards.
        devices: A sequence of target devices.

    Returns:
        A sharded array constructed from the inputs.
    """
    from zero_jax.numpy import stack

    return stack(shards)
