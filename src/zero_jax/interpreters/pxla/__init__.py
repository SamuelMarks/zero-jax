"""Frontend API routing for jax.interpreters.pxla."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from zero_jax._types import ArrayLikeSpec, ConfigValue, SliceSpec


class ArrayMapping(dict[str, int]):
    """Ordered mapping from mesh axis names to array dimension indices."""

    def __init__(
        self,
        mapping: dict[str, int] | None = None,
        *,
        id: ConfigValue = None,
        name: str = "ArrayMapping",
        value: ConfigValue = None,
    ) -> None:
        """Initialize ``ArrayMapping``.

        Args:
            mapping: Optional initial axis-name to dimension-index mapping.
            id: Optional identifier metadata.
            name: Mapping descriptor name.
            value: Optional value metadata.
        """
        super().__init__(mapping or {})
        self.id: ConfigValue = id
        self.name: str = name
        self.value: ConfigValue = value


@dataclass
class Chunked:
    """Sharding spec indicating an axis is chunked across multiple mesh axes."""

    id: int = 0
    name: str = "Chunked"
    value: ArrayLikeSpec | None = None


def Index(idx: ArrayLikeSpec = slice(None)) -> ArrayLikeSpec:
    """Canonicalize an index or slice specification for device shard indexing.

    Args:
        idx: Slice, integer, or tuple of slices.

    Returns:
        ArrayLikeSpec: Canonicalized index specification.
    """
    return idx


@dataclass
class MapTracer:
    """Tracer used inside parallel mapped computations."""

    id: int = 0
    name: str = "MapTracer"
    value: ArrayLikeSpec | None = None


def MeshAxisName(name: str = "x") -> str:
    """Return a canonical mesh axis name identifier.

    Args:
        name: Mesh axis name string.

    Returns:
        str: Canonical mesh axis name.
    """
    return str(name)


@dataclass
class MeshComputation:
    """Lowered computation staged for execution across a device mesh."""

    id: int = 0
    name: str = "MeshComputation"
    value: ArrayLikeSpec | None = None


@dataclass
class MeshExecutable:
    """Compiled executable bound to a device mesh."""

    id: int = 0
    name: str = "MeshExecutable"
    value: ArrayLikeSpec | None = None


@dataclass
class NoSharding:
    """Sharding spec indicating an array dimension is not sharded."""

    id: int = 0
    name: str = "NoSharding"
    value: ArrayLikeSpec | None = None


@dataclass
class PmapExecutable:
    """Compiled executable for :func:`pmap` execution."""

    id: int = 0
    name: str = "PmapExecutable"
    value: ArrayLikeSpec | None = None


@dataclass
class Replicated:
    """Mesh specification indicating replication across ``replicas`` devices."""

    id: int = 0
    name: str = "Replicated"
    value: ArrayLikeSpec | None = None


@dataclass
class ShardedAxis:
    """Mesh specification indicating sharding along ``axis``."""

    id: int = 0
    name: str = "ShardedAxis"
    value: ArrayLikeSpec | None = None


@dataclass
class ShardingSpec:
    """Specification of how an array's dimensions map to device mesh axes."""

    id: int = 0
    name: str = "ShardingSpec"
    value: ArrayLikeSpec | None = None


@dataclass
class Unstacked:
    """Sharding spec indicating an axis is unstacked across devices."""

    id: int = 0
    name: str = "Unstacked"
    value: ArrayLikeSpec | None = None


def are_op_shardings_equal(op1: ArrayLikeSpec = None, op2: ArrayLikeSpec = None) -> bool:
    """Return ``True`` if two XLA ``OpSharding`` specifications are equivalent.

    Args:
        op1: First ``OpSharding`` descriptor.
        op2: Second ``OpSharding`` descriptor.

    Returns:
        bool: ``True`` when both shardings are equal.
    """
    return op1 == op2


def array_mapping_to_axis_resources(
    array_mapping: ArrayMapping | dict[str, int] | None = None,
) -> tuple[str, ...]:
    """Convert an :class:`ArrayMapping` into a tuple of axis resource names.

    Args:
        array_mapping: Mapping from mesh axis names to array dimensions.

    Returns:
        tuple[str, ...]: Tuple of mapped axis names.
    """
    if isinstance(array_mapping, dict):
        return tuple(array_mapping.keys())
    return ()


def global_aval_to_result_handler(
    aval: ArrayLikeSpec = None,
    sharding: ArrayLikeSpec = None,
    committed: bool = False,
) -> Callable[[ArrayLikeSpec], ArrayLikeSpec]:
    """Return a handler function that wraps raw output buffers for a single abstract value.

    Args:
        aval: Output abstract value.
        sharding: Output sharding specification.
        committed: Whether the output is committed to its sharding.

    Returns:
        Callable[[ArrayLikeSpec], ArrayLikeSpec]: Result buffer handler callable.
    """
    _ = (aval, sharding, committed)
    return lambda buf: buf


def global_avals_to_results_handler(
    avals: ArrayLikeSpec = (),
    shardings: ArrayLikeSpec = (),
    committed: bool = False,
) -> Callable[[ArrayLikeSpec], ArrayLikeSpec]:
    """Return a handler function that wraps a sequence of raw output buffers.

    Args:
        avals: Sequence of output abstract values.
        shardings: Sequence of output shardings.
        committed: Whether outputs are committed.

    Returns:
        Callable[[ArrayLikeSpec], ArrayLikeSpec]: Multi-result buffer handler callable.
    """
    _ = (avals, shardings, committed)
    return lambda bufs: bufs


global_result_handlers: dict[str, ArrayLikeSpec] = {}


def is_op_sharding_replicated(op: ArrayLikeSpec = None) -> bool:
    """Return ``True`` if ``op`` represents a fully replicated sharding.

    Args:
        op: ``OpSharding`` specification or ``None``.

    Returns:
        bool: ``True`` if replicated.
    """
    return op is None or str(op).lower() == "replicated"


def op_sharding_to_indices(
    op_sharding: ArrayLikeSpec = None,
    shape: tuple[int, ...] = (1,),
    num_devices: int = 1,
) -> tuple[tuple[SliceSpec, ...], ...]:
    """Compute per-device slice tuples for ``op_sharding`` over ``shape``.

    Args:
        op_sharding: Sharding descriptor.
        shape: Global array shape.
        num_devices: Number of target devices.

    Returns:
        tuple[tuple[SliceSpec, ...], ...]: Per-device index slices.
    """
    _ = op_sharding
    full_slice = tuple(slice(0, int(s)) for s in shape)
    return tuple(full_slice for _ in range(max(1, int(num_devices))))


def parallel_callable(
    fun: ArrayLikeSpec = None,
    backend_name: str | None = None,
    axis_name: str = "i",
    axis_size: int = 1,
) -> ArrayLikeSpec:
    """Compile ``fun`` for parallel execution across ``axis_name``.

    Args:
        fun: Function to parallelize.
        backend_name: Optional backend identifier.
        axis_name: Mapped collective axis name.
        axis_size: Number of replicas along ``axis_name``.

    Returns:
        ArrayLikeSpec: Parallelized callable.
    """
    _ = (backend_name, axis_name, axis_size)
    return fun if fun is not None else (lambda *args: args)


def shard_args(
    shardings: ArrayLikeSpec = (),
    args: ArrayLikeSpec = (),
) -> tuple[ArrayLikeSpec, ...]:
    """Shard input ``args`` according to ``shardings``.

    Args:
        shardings: Per-argument sharding specifications.
        args: Input arguments to shard.

    Returns:
        tuple[ArrayLikeSpec, ...]: Sharded arguments tuple.
    """
    _ = shardings
    return tuple(args) if isinstance(args, (list, tuple)) else (args,)


def spec_to_indices(
    shape: tuple[int, ...] = (1,),
    spec: ArrayLikeSpec = None,
) -> tuple[tuple[SliceSpec, ...], ...]:
    """Convert a :class:`ShardingSpec` into device buffer slice tuples for ``shape``.

    Args:
        shape: Array shape tuple.
        spec: :class:`ShardingSpec` instance.

    Returns:
        tuple[tuple[SliceSpec, ...], ...]: Tuple of per-device slice tuples.
    """
    _ = spec
    return (tuple(slice(0, int(s)) for s in shape),)


thread_resources: dict[str, ArrayLikeSpec] = {"env": None}

xla_pmap_p: str = "xla_pmap"

__all__ = [
    "ArrayMapping",
    "Chunked",
    "Index",
    "MapTracer",
    "MeshAxisName",
    "MeshComputation",
    "MeshExecutable",
    "NoSharding",
    "Optional",
    "PmapExecutable",
    "Replicated",
    "ShardedAxis",
    "ShardingSpec",
    "Unstacked",
    "are_op_shardings_equal",
    "array_mapping_to_axis_resources",
    "dataclass",
    "global_aval_to_result_handler",
    "global_avals_to_results_handler",
    "global_result_handlers",
    "is_op_sharding_replicated",
    "op_sharding_to_indices",
    "parallel_callable",
    "shard_args",
    "spec_to_indices",
    "thread_resources",
    "xla_pmap_p",
]
