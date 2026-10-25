"""Frontend API routing for jax.sharding."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from types import TracebackType

from ml_switcheroo_ir import LogicalMesh as _LogicalMesh
from ml_switcheroo_ir import PartitionSpec as _IRPartitionSpec

from zero_jax._types import ConfigValue, PyTree, VariadicCallable


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
        _ = ndim
        return type(self) is type(other) and self.device_set == other.device_set


class PartitionSpec(Sharding):
    """Tuple-like specification describing how array dimensions map to mesh axes."""

    ir_partition_spec_cls = _IRPartitionSpec

    def __init__(
        self,
        *args: str | tuple[str, ...] | None,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize ``PartitionSpec`` with per-dimension axis names.

        Args:
            *args: Axis names (or ``None``) for each tensor dimension.
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
            **kwargs: Additional keyword metadata.
        """
        super().__init__(id=id, name=name, value=value)
        self._partitions: tuple[str | tuple[str, ...] | None, ...] = tuple(args)
        self.kwargs = kwargs

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
        axis_types: Sequence[ConfigValue] | None = None,
        size: int | None = None,
        _cached_device_list: Sequence[ConfigValue] | None = None,
        *args: ConfigValue,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize ``Mesh`` with devices and axis names.

        Args:
            devices: Sequence of device objects or identifiers.
            axis_names: Sequence of mesh axis names.
            axis_types: Optional per-axis :class:`AxisType` specifications.
            size: Optional total device count override.
            _cached_device_list: Optional cached flat device list.
            *args: Additional positional arguments.
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(devices=devices, id=id, name=name, value=value)
        self.devices = tuple(devices)
        self.axis_names = tuple(axis_names)
        self.axis_types = tuple(axis_types) if axis_types is not None else None
        self.size = size if size is not None else len(self.devices)
        self._cached_device_list = _cached_device_list
        self.args = args
        self.kwargs = kwargs

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
        *,
        memory_kind: str | None = None,
        _logical_device_ids: tuple[int, ...] | None = None,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize ``NamedSharding``.

        Args:
            mesh: The target :class:`Mesh`.
            spec: The :class:`PartitionSpec` describing dimension partitioning.
            memory_kind: Optional target memory kind (e.g. ``"device"`` or ``"pinned_host"``).
            _logical_device_ids: Optional logical device ID permutation.
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
        """
        resolved_mesh = mesh if isinstance(mesh, Mesh) else Mesh()
        resolved_spec = spec if isinstance(spec, PartitionSpec) else PartitionSpec()
        super().__init__(devices=resolved_mesh.devices, id=id, name=name, value=value)
        self.mesh = resolved_mesh
        self.spec = resolved_spec
        self.memory_kind = memory_kind
        self._logical_device_ids = _logical_device_ids

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
        *,
        memory_kind: str | None = None,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize ``SingleDeviceSharding``.

        Args:
            device: Target device.
            memory_kind: Optional memory kind string.
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
        """
        super().__init__(devices=(device,), id=id, name=name, value=value)
        self.device = device
        self.memory_kind = memory_kind


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


class AbstractDevice:
    """Abstract description of a logical device without concrete runtime handles."""

    def __init__(
        self,
        device_kind: str = "cpu",
        num_cores: int | None = 1,
        platform: str = "cpu",
    ) -> None:
        """Initialize ``AbstractDevice``.

        Args:
            device_kind: Kind of hardware device (e.g. ``"cpu"``, ``"tpu"``).
            num_cores: Optional number of cores on the device.
            platform: Platform name string.
        """
        self.device_kind = device_kind
        self.num_cores = num_cores
        self.platform = platform


class AbstractMesh:
    """Abstract device mesh specification defined solely by axis sizes and names."""

    def __init__(
        self,
        axis_sizes: tuple[int, ...] = (),
        axis_names: tuple[str, ...] = (),
        axis_types: ConfigValue = None,
        abstract_device: AbstractDevice | None = None,
        size: int | None = None,
        *args: ConfigValue,
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize ``AbstractMesh``.

        Args:
            axis_sizes: Tuple of integer sizes for each mesh axis.
            axis_names: Tuple of string names for each mesh axis.
            axis_types: Optional per-axis :class:`AxisType` specifications.
            abstract_device: Optional :class:`AbstractDevice` descriptor.
            size: Optional total mesh size override.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.axis_sizes = tuple(axis_sizes)
        self.axis_names = tuple(axis_names)
        self.axis_types = axis_types
        self.abstract_device = abstract_device
        import math

        self.size = size if size is not None else (int(math.prod(self.axis_sizes)) if self.axis_sizes else 1)
        self.args = args
        self.kwargs = kwargs


class AxisType:
    """Enumeration of mesh axis partitioning modes."""

    Auto = "Auto"
    Explicit = "Explicit"
    Manual = "Manual"

    def __init__(
        self,
        Auto: str = "Auto",
        Explicit: str = "Explicit",
        Manual: str = "Manual",
        *args: ConfigValue,
        **kwds: ConfigValue,
    ) -> None:
        """Initialize ``AxisType``.

        Args:
            Auto: Automatic compiler partitioning mode.
            Explicit: Explicit type-level partitioning mode.
            Manual: Manual per-shard partitioning mode.
            *args: Additional positional arguments.
            **kwds: Additional keyword arguments.
        """
        self.Auto = Auto
        self.Explicit = Explicit
        self.Manual = Manual
        self.args = args
        self.kwds = kwds


class ManualAxisType:
    """Configuration descriptor for manual mesh axes."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize ``ManualAxisType``.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


class UnreducedKind:
    """Enumeration of unreduced shard reduction kinds."""

    sum = "sum"
    max = "max"
    min = "min"

    def __init__(
        self,
        sum: str = "sum",
        max: str = "max",
        min: str = "min",
        *args: ConfigValue,
        **kwds: ConfigValue,
    ) -> None:
        """Initialize ``UnreducedKind``.

        Args:
            sum: Unreduced sum reduction kind.
            max: Unreduced max reduction kind.
            min: Unreduced min reduction kind.
            *args: Additional positional arguments.
            **kwds: Additional keyword arguments.
        """
        self.sum = sum
        self.max = max
        self.min = min
        self.args = args
        self.kwds = kwds


_ACTIVE_MESH: Mesh | None = None
_ACTIVE_ABSTRACT_MESH: AbstractMesh | None = None


class set_mesh:
    """Context manager and setter for the thread-local active :class:`Mesh`."""

    def __init__(self, mesh: Mesh | None) -> None:
        """Initialize ``set_mesh`` and activate ``mesh``.

        Args:
            mesh: The :class:`Mesh` to set as active, or ``None`` to clear.
        """
        global _ACTIVE_MESH
        self.prev_mesh = _ACTIVE_MESH
        self.prev_abstract_mesh = _ACTIVE_ABSTRACT_MESH
        self.mesh = mesh
        _ACTIVE_MESH = mesh

    def __enter__(self) -> Mesh | None:
        """Enter the mesh context.

        Returns:
            Mesh | None: The active mesh.
        """
        return self.mesh

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Restore the previous mesh upon exiting the context.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception instance if raised.
            exc_tb: Exception traceback if raised.
        """
        _ = (exc_type, exc_val, exc_tb)
        global _ACTIVE_MESH
        _ACTIVE_MESH = self.prev_mesh


class use_abstract_mesh:
    """Context manager for temporarily activating an :class:`AbstractMesh`."""

    def __init__(self, mesh: AbstractMesh) -> None:
        """Initialize ``use_abstract_mesh``.

        Args:
            mesh: The :class:`AbstractMesh` to activate inside the context.
        """
        global _ACTIVE_ABSTRACT_MESH
        self.prev = _ACTIVE_ABSTRACT_MESH
        self.mesh = mesh
        _ACTIVE_ABSTRACT_MESH = mesh

    def __enter__(self) -> AbstractMesh:
        """Enter the abstract mesh context.

        Returns:
            AbstractMesh: Active abstract mesh.
        """
        return self.mesh

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Restore the previous abstract mesh upon exiting the context.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception instance if raised.
            exc_tb: Exception traceback if raised.
        """
        _ = (exc_type, exc_val, exc_tb)
        global _ACTIVE_ABSTRACT_MESH
        _ACTIVE_ABSTRACT_MESH = self.prev


def get_mesh() -> Mesh:
    """Return the currently active concrete :class:`Mesh` (or a default 1-device mesh).

    Returns:
        Mesh: Currently active :class:`Mesh`.
    """
    return _ACTIVE_MESH if _ACTIVE_MESH is not None else Mesh()


def get_abstract_mesh() -> AbstractMesh:
    """Return the currently active :class:`AbstractMesh` (or an empty ``AbstractMesh``).

    Returns:
        AbstractMesh: Currently active :class:`AbstractMesh`.
    """
    return _ACTIVE_ABSTRACT_MESH if _ACTIVE_ABSTRACT_MESH is not None else AbstractMesh()


def auto_axes(
    f: VariadicCallable[PyTree] | None = None,
    *,
    axes: str | tuple[str, ...] | None = None,
    out_sharding: ConfigValue = None,
) -> VariadicCallable[PyTree]:
    """Decorate a function to execute with automatic axis partitioning.

    Args:
        f: Function to wrap.
        axes: Axis name or tuple of axis names switched to ``Auto`` mode.
        out_sharding: Optional output sharding specification.

    Returns:
        VariadicCallable[PyTree]: Wrapped function or decorator.
    """
    _ = (axes, out_sharding)
    if f is None:

        def _decorator(fn: VariadicCallable[PyTree]) -> VariadicCallable[PyTree]:
            """Decorate ``fn`` with automatic axes.

            Args:
                fn: Target callable.

            Returns:
                VariadicCallable[PyTree]: Wrapped callable.
            """
            return fn

        return _decorator
    return f


def explicit_axes(
    f: VariadicCallable[PyTree] | None = None,
    *,
    axes: str | tuple[str, ...] | None = None,
    in_sharding: ConfigValue = None,
) -> VariadicCallable[PyTree]:
    """Decorate a function to execute with explicit axis partitioning.

    Args:
        f: Function to wrap.
        axes: Axis name or tuple of axis names switched to ``Explicit`` mode.
        in_sharding: Optional input sharding specification.

    Returns:
        VariadicCallable[PyTree]: Wrapped function or decorator.
    """
    _ = (axes, in_sharding)
    if f is None:

        def _decorator(fn: VariadicCallable[PyTree]) -> VariadicCallable[PyTree]:
            """Decorate ``fn`` with explicit axes.

            Args:
                fn: Target callable.

            Returns:
                VariadicCallable[PyTree]: Wrapped callable.
            """
            return fn

        return _decorator
    return f


def make_single_device_sharding(
    device: ConfigValue = "cpu:0",
    *,
    memory_kind: str | None = None,
) -> SingleDeviceSharding:
    """Construct a :class:`SingleDeviceSharding` for ``device``.

    Args:
        device: Target device.
        memory_kind: Optional memory kind string.

    Returns:
        SingleDeviceSharding: Constructed single-device sharding.
    """
    return SingleDeviceSharding(device=device, memory_kind=memory_kind)


@contextlib.contextmanager
def no_collectives(new_val: bool = True) -> Iterator[bool]:
    """Context manager forbidding collective operations inside its dynamic scope.

    Args:
        new_val: Whether collectives are disallowed.

    Yields:
        bool: Active flag state.
    """
    yield new_val


def reshard(xs: PyTree, out_shardings: ConfigValue) -> PyTree:
    """Reshard a PyTree of arrays ``xs`` to match ``out_shardings``.

    Args:
        xs: Input array or PyTree of arrays.
        out_shardings: Target sharding specification PyTree.

    Returns:
        PyTree: Resharded ``xs``.
    """
    _ = out_shardings
    return xs


__all__ = [
    "AbstractDevice",
    "AbstractMesh",
    "AxisType",
    "GSPMDSharding",
    "ManualAxisType",
    "Mesh",
    "NamedSharding",
    "PartitionSpec",
    "PmapSharding",
    "PositionalSharding",
    "Sharding",
    "SingleDeviceSharding",
    "UnreducedKind",
    "auto_axes",
    "explicit_axes",
    "get_abstract_mesh",
    "get_mesh",
    "make_single_device_sharding",
    "no_collectives",
    "reshard",
    "set_mesh",
    "use_abstract_mesh",
]
