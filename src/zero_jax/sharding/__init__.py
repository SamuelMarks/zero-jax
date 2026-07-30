"""Frontend API routing for jax.sharding."""

from typing import Any


class Sharding:
    """Base class for all sharding types."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class GSPMDSharding:
    """Frontend state holder for GSPMDSharding."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class Mesh:
    """Declare the hardware resources available in the scope of this manager."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class NamedSharding:
    """A :class:`NamedSharding` expresses sharding using named axes."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class PartitionSpec:
    """Tuple describing how to partition an array across a mesh of devices."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class PmapSharding:
    """Describes a sharding used by :func:`jax.pmap`."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class PositionalSharding:
    """Frontend state holder for PositionalSharding."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover


class SingleDeviceSharding:
    """A :class:`Sharding` that places its data on a single device."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)  # pragma: no cover
