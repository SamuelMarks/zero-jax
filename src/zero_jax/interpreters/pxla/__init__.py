"""Frontend API routing for jax.interpreters.pxla."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue


class ArrayMapping:
    """Dictionary that remembers insertion order."""

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.id = id
        self.name = name
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs


@dataclass
class Chunked:
    """Data structure for Chunked.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "Chunked"
    value: ArrayLikeSpec | None = None


def Index(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for Index.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.Index(*args, **kwargs)


@dataclass
class MapTracer:
    """Data structure for MapTracer.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "MapTracer"
    value: ArrayLikeSpec | None = None


def MeshAxisName(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Special type indicating an unconstrained type.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.MeshAxisName(*args, **kwargs)


@dataclass
class MeshComputation:
    """Data structure for MeshComputation.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "MeshComputation"
    value: ArrayLikeSpec | None = None


@dataclass
class MeshExecutable:
    """Data structure for MeshExecutable.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "MeshExecutable"
    value: ArrayLikeSpec | None = None


@dataclass
class NoSharding:
    """Data structure for NoSharding.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "NoSharding"
    value: ArrayLikeSpec | None = None


@dataclass
class PmapExecutable:
    """Data structure for PmapExecutable.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "PmapExecutable"
    value: ArrayLikeSpec | None = None


@dataclass
class Replicated:
    """Data structure for Replicated.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "Replicated"
    value: ArrayLikeSpec | None = None


@dataclass
class ShardedAxis:
    """Data structure for ShardedAxis.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "ShardedAxis"
    value: ArrayLikeSpec | None = None


@dataclass
class ShardingSpec:
    """Data structure for ShardingSpec.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "ShardingSpec"
    value: ArrayLikeSpec | None = None


@dataclass
class Unstacked:
    """Data structure for Unstacked.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "Unstacked"
    value: ArrayLikeSpec | None = None


def are_op_shardings_equal(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for are_op_shardings_equal.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.are_op_shardings_equal(*args, **kwargs)


def array_mapping_to_axis_resources(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for array_mapping_to_axis_resources.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.array_mapping_to_axis_resources(*args, **kwargs)


def global_aval_to_result_handler(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Returns a function for handling the raw buffers of a single output aval.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.global_aval_to_result_handler(*args, **kwargs)


def global_avals_to_results_handler(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for global_avals_to_results_handler.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.global_avals_to_results_handler(*args, **kwargs)


global_result_handlers: ArrayLikeSpec = None


def is_op_sharding_replicated(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for is_op_sharding_replicated.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.is_op_sharding_replicated(*args, **kwargs)


def op_sharding_to_indices(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for op_sharding_to_indices.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.op_sharding_to_indices(*args, **kwargs)


def parallel_callable(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for parallel_callable.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.parallel_callable(*args, **kwargs)


def shard_args(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for shard_args.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.shard_args(*args, **kwargs)


def spec_to_indices(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Returns numpy-style indices corresponding to a sharding spec.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.spec_to_indices(*args, **kwargs)


thread_resources: ArrayLikeSpec = None

xla_pmap_p: ArrayLikeSpec = None
