"""Core JAX API structures and classes."""

from __future__ import annotations

from collections.abc import Sequence

from zero_jax._config_internal import config as _internal_config
from zero_jax._types import ArrayLikeSpec, ConfigValue, DTypeSpec, IndexSpec, ShapeTuple, T
from zero_jax.numpy.lax_numpy import ndarray
from zero_jax.sharding import NamedSharding, Sharding, SingleDeviceSharding
from zero_jax.tree_util import tree_map


class Device:
    """Represents a logical computational device.

    Attributes:
        platform: The target execution platform name (e.g. ``"cpu"``).
        id: Logical device index.
    """

    def __init__(self, platform: str = "cpu", id: int = 0) -> None:
        """Initialize a ``Device`` instance.

        Args:
            platform: Platform identifier string.
            id: Logical integer device identifier.
        """
        self.platform = platform
        self.device_kind = platform
        self.id = id
        self.process_index = 0
        self.host_id = 0

    def __eq__(self, other: ConfigValue) -> bool:
        """Compare two ``Device`` instances for equality.

        Args:
            other: Comparison target.

        Returns:
            bool: ``True`` if platform and id match.
        """
        return isinstance(other, Device) and self.platform == other.platform and self.id == other.id

    def __hash__(self) -> int:
        """Return hash of the ``Device`` instance.

        Returns:
            int: Hash value.
        """
        return hash((self.platform, self.id))

    def __repr__(self) -> str:
        """Return canonical string representation of this device.

        Returns:
            str: Formatted device string.
        """
        return f"{self.platform}:{self.id}"


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
        sharding: Sharding | NamedSharding | None = None,
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

    def __init__(self, device: Device | str | None, index: IndexSpec) -> None:
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
    return Device("cpu")


def device_count(backend: str | None = None) -> int:
    """Return the total number of devices available.

    Synchronized with ``jax_host_device_count`` and ``zero_chex.set_n_cpu_devices``.

    Args:
        backend: Optional backend name.

    Returns:
        int: Total number of devices.
    """
    _ = backend
    raw = _internal_config.read("jax_host_device_count")
    return max(1, int(raw)) if isinstance(raw, int) else 1


def local_device_count(backend: str | None = None) -> int:
    """Return the number of local devices available to the current process.

    Args:
        backend: Optional backend name.

    Returns:
        int: Number of local devices.
    """
    return device_count(backend=backend)


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


def device_put(x: T, device: Device | Sharding | str | None = None) -> T:
    """Transfer data to the specified device or sharding specification.

    Args:
        x: The array or PyTree of arrays to transfer.
        device: Optional target device or :class:`Sharding`.

    Returns:
        T: The transferred value with sharding metadata attached when ``device`` is provided.
    """
    if device is None:
        return x
    resolved_sharding = device if isinstance(device, Sharding) else SingleDeviceSharding(device)

    def _put_leaf(leaf: ConfigValue) -> ConfigValue:
        """Attach ``resolved_sharding`` to an ``ndarray`` leaf.

        Args:
            leaf: Input PyTree leaf.

        Returns:
            ConfigValue: Leaf with updated sharding metadata.
        """
        if isinstance(leaf, ndarray):
            return ndarray(leaf._tensor, sharding=resolved_sharding)
        return leaf

    return tree_map(_put_leaf, x)


def device_put_replicated(x: T, devices: Sequence[Device | str | None]) -> list[T]:
    """Replicate data across the specified sequence of devices.

    Args:
        x: The array or PyTree of arrays to replicate.
        devices: Sequence of target devices.

    Returns:
        list[T]: List of replicated values per device.
    """
    return [x for _ in devices]


def device_put_sharded(shards: Sequence[ArrayLikeSpec], devices: Sequence[Device | str | None]) -> ndarray:
    """Construct an array from a sequence of shards on the specified devices.

    Args:
        shards: Sequence of arrays representing the shards.
        devices: Sequence of target devices.

    Returns:
        ndarray: Stacked array constructed from ``shards``.
    """
    from zero_jax.numpy import stack
    from zero_jax.sharding import PositionalSharding

    stacked = stack(shards)
    return ndarray(stacked._tensor, sharding=PositionalSharding(tuple(devices)))


__all__ = [
    "Device",
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
