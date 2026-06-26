"""Frontend API routing for jax.sharding."""

from typing import Any


class GSPMDSharding:
    """Mock implementation for GSPMDSharding."""

    pass


class Mesh:
    """Declare the hardware resources available in the scope of this manager."""

    pass


class NamedSharding:
    """A :class:`NamedSharding` expresses sharding using named axes."""

    pass


class PartitionSpec:
    """Tuple describing how to partition an array across a mesh of devices."""

    pass


class PmapSharding:
    """Describes a sharding used by :func:`jax.pmap`."""

    pass


class PositionalSharding:
    """Mock implementation for PositionalSharding."""

    pass


class SingleDeviceSharding:
    """A :class:`Sharding` that places its data on a single device."""

    pass
