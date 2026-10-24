"""Frontend API routing for jax.sharding."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_ir import LogicalMesh as _LogicalMesh
from ml_switcheroo_ir import PartitionSpec as _IRPartitionSpec

from zero_jax._types import ConfigValue


class Sharding:
    """Base class for all JAX sharding specifications."""

    def __init__(
        self,
        devices: Sequence[ConfigValue] | None = None,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize ``Sharding``.

        Args:
            devices: Optional sequence of devices covered by this sharding.
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
        """
        self._devices = tuple(devices) if devices is not None else ("cpu:0",)
        self.id = id
        self.name = name
        self.value = value

    @property
    def device_set(self) -> set[ConfigValue]:
        """Return the set of devices referenced by this sharding.

        Returns:
            set[ConfigValue]: Set of devices.
        """
        return set(self._devices)

    @property
    def is_fully_addressable(self) -> bool:
        """Return whether all devices in this sharding are locally addressable.

        Returns:
            bool: Always ``True`` in local execution.
        """
        return True

    @property
    def is_fully_replicated(self) -> bool:
        """Return whether array data is replicated across all devices.

        Returns:
            bool: ``True`` if only a single device is present or spec is unpartitioned.
        """
        return len(self._devices) <= 1

    def is_equivalent_to(self, other: Sharding, ndim: int = 0) -> bool:
        """Check whether two sharding instances are equivalent for a given rank.

        Args:
            other: Another ``Sharding`` instance.
            ndim: Rank of the partitioned array.

        Returns:
            bool: ``True`` if equivalent.
        """
        return type(self) is type(other) and self.device_set == other.device_set


class PartitionSpec(Sharding):
    """Tuple-like specification describing how array dimensions map to mesh axes."""

    ir_partition_spec_cls = _IRPartitionSpec

    def __init__(
        self,
        *partitions: str | tuple[str, ...] | None,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize ``PartitionSpec`` with per-dimension axis names.

        Args:
            *partitions: Axis names (or ``None``) for each tensor dimension.
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
        """
        super().__init__(id=id, name=name, value=value)
        self._partitions: tuple[str | tuple[str, ...] | None, ...] = tuple(partitions)

    def __len__(self) -> int:
        """Return the number of partitioned dimensions.

        Returns:
            int: Length of the partition tuple.
        """
        return len(self._partitions)

    def __getitem__(self, idx: int) -> str | tuple[str, ...] | None:
        """Return the partition axis specification at ``idx``.

        Args:
            idx: Dimension index.

        Returns:
            str | tuple[str, ...] | None: Axis specification.
        """
        return self._partitions[idx]

    def __eq__(self, other: ConfigValue) -> bool:
        """Compare two ``PartitionSpec`` objects for equality.

        Args:
            other: Comparison target.

        Returns:
            bool: ``True`` if both represent the same partition tuple.
        """
        if isinstance(other, PartitionSpec):
            return self._partitions == other._partitions
        return False


class Mesh(Sharding):
    """Declare hardware device resources organized as a named multidimensional grid."""

    ir_mesh_cls = _LogicalMesh

    def __init__(
        self,
        devices: Sequence[ConfigValue] = ("cpu:0",),
        axis_names: Sequence[str] = ("x",),
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize ``Mesh`` with devices and axis names.

        Args:
            devices: Sequence of device objects or identifiers.
            axis_names: Sequence of mesh axis names.
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
        """
        super().__init__(devices=devices, id=id, name=name, value=value)
        self.devices = tuple(devices)
        self.axis_names = tuple(axis_names)

    @property
    def shape(self) -> dict[str, int]:
        """Return a mapping from axis names to axis sizes.

        Returns:
            dict[str, int]: Axis name to size mapping.
        """
        if not self.axis_names:
            return {}
        base = max(1, len(self.devices) // len(self.axis_names))
        return dict.fromkeys(self.axis_names, base)


class NamedSharding(Sharding):
    """Express array sharding using a :class:`Mesh` and :class:`PartitionSpec`."""

    def __init__(
        self,
        mesh: Mesh | ConfigValue = None,
        spec: PartitionSpec | ConfigValue = None,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize ``NamedSharding``.

        Args:
            mesh: The target :class:`Mesh`.
            spec: The :class:`PartitionSpec` describing dimension partitioning.
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
        """
        resolved_mesh = mesh if isinstance(mesh, Mesh) else Mesh()
        resolved_spec = spec if isinstance(spec, PartitionSpec) else PartitionSpec()
        super().__init__(devices=resolved_mesh.devices, id=id, name=name, value=value)
        self.mesh = resolved_mesh
        self.spec = resolved_spec

    @property
    def is_fully_replicated(self) -> bool:
        """Return whether the ``PartitionSpec`` has no partitioned axes.

        Returns:
            bool: ``True`` if all spec entries are ``None`` or empty.
        """
        return all(p is None for p in self.spec._partitions)


class SingleDeviceSharding(Sharding):
    """A :class:`Sharding` that places all data on a single device."""

    def __init__(
        self,
        device: ConfigValue = "cpu:0",
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize ``SingleDeviceSharding``.

        Args:
            device: Target device.
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
        """
        super().__init__(devices=(device,), id=id, name=name, value=value)
        self.device = device


class PositionalSharding(Sharding):
    """Sharding that specifies device placement by positional shape grid."""

    def __init__(
        self,
        devices: Sequence[ConfigValue] = ("cpu:0",),
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize ``PositionalSharding``.

        Args:
            devices: Sequence of target devices.
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
        """
        super().__init__(devices=devices, id=id, name=name, value=value)

    def replicate(self) -> PositionalSharding:
        """Return a replicated ``PositionalSharding``.

        Returns:
            PositionalSharding: Replicated sharding instance.
        """
        return PositionalSharding(devices=self._devices, id=self.id, name=self.name, value=self.value)


class PmapSharding(Sharding):
    """Describe a sharding used by :func:`jax.pmap`."""

    def __init__(
        self,
        devices: Sequence[ConfigValue] = ("cpu:0",),
        sharding_spec: ConfigValue = None,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize ``PmapSharding``.

        Args:
            devices: Sequence of pmap devices.
            sharding_spec: Optional sharding specification.
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
        """
        super().__init__(devices=devices, id=id, name=name, value=value)
        self.devices = tuple(devices)
        self.sharding_spec = sharding_spec


class GSPMDSharding(Sharding):
    """Low-level GSPMD sharding specification."""

    def __init__(
        self,
        devices: Sequence[ConfigValue] = ("cpu:0",),
        op_sharding: ConfigValue = None,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize ``GSPMDSharding``.

        Args:
            devices: Sequence of devices.
            op_sharding: Optional HLO OpSharding representation.
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
        """
        super().__init__(devices=devices, id=id, name=name, value=value)
        self.op_sharding = op_sharding


__all__ = [
    "GSPMDSharding",
    "Mesh",
    "NamedSharding",
    "PartitionSpec",
    "PmapSharding",
    "PositionalSharding",
    "Sharding",
    "SingleDeviceSharding",
]
