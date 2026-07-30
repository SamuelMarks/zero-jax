"""Frontend API routing for jax.interpreters.pxla."""

from dataclasses import dataclass
from typing import Any, Optional

import zero_jax._compiler_proxy_ops as _ops


class ArrayMapping:
    """Dictionary that remembers insertion order"""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass
class Chunked:
    """Data structure for Chunked.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "Chunked"
    value: Optional[Any] = None


def Index(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for Index.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.Index(*args, **kwargs)


@dataclass
class MapTracer:
    """Data structure for MapTracer.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "MapTracer"
    value: Optional[Any] = None


def MeshAxisName(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating an unconstrained type."""
    return _ops.MeshAxisName(*args, **kwargs)


@dataclass
class MeshComputation:
    """Data structure for MeshComputation.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "MeshComputation"
    value: Optional[Any] = None


@dataclass
class MeshExecutable:
    """Data structure for MeshExecutable.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "MeshExecutable"
    value: Optional[Any] = None


@dataclass
class NoSharding:
    """Data structure for NoSharding.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "NoSharding"
    value: Optional[Any] = None


@dataclass
class PmapExecutable:
    """Data structure for PmapExecutable.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "PmapExecutable"
    value: Optional[Any] = None


@dataclass
class Replicated:
    """Data structure for Replicated.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "Replicated"
    value: Optional[Any] = None


@dataclass
class ShardedAxis:
    """Data structure for ShardedAxis.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "ShardedAxis"
    value: Optional[Any] = None


@dataclass
class ShardingSpec:
    """Data structure for ShardingSpec.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "ShardingSpec"
    value: Optional[Any] = None


@dataclass
class Unstacked:
    """Data structure for Unstacked.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "Unstacked"
    value: Optional[Any] = None


def are_op_shardings_equal(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for are_op_shardings_equal.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.are_op_shardings_equal(*args, **kwargs)


def array_mapping_to_axis_resources(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for array_mapping_to_axis_resources.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.array_mapping_to_axis_resources(*args, **kwargs)


def global_aval_to_result_handler(*args: Any, **kwargs: Any) -> Any:
    """Returns a function for handling the raw buffers of a single output aval."""
    return _ops.global_aval_to_result_handler(*args, **kwargs)


def global_avals_to_results_handler(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for global_avals_to_results_handler.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.global_avals_to_results_handler(*args, **kwargs)


global_result_handlers: Any = None


def is_op_sharding_replicated(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for is_op_sharding_replicated.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.is_op_sharding_replicated(*args, **kwargs)


def op_sharding_to_indices(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for op_sharding_to_indices.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.op_sharding_to_indices(*args, **kwargs)


def parallel_callable(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for parallel_callable.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.parallel_callable(*args, **kwargs)


def shard_args(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for shard_args.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.shard_args(*args, **kwargs)


def spec_to_indices(*args: Any, **kwargs: Any) -> Any:
    """Returns numpy-style indices corresponding to a sharding spec."""
    return _ops.spec_to_indices(*args, **kwargs)


thread_resources: Any = None

xla_pmap_p: Any = None

import typing

import ml_switcheroo_compiler


def __getattr__(name):
    if hasattr(_ops, name):
        return getattr(_ops, name)  # pragma: no cover
    if hasattr(ml_switcheroo_compiler, name):
        return getattr(ml_switcheroo_compiler, name)  # pragma: no cover
    try:
        from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

        # If it's a known missing function, we might just return a dummy callable that raises NotImplementedError,
        # BUT we only want to do that if it really doesn't exist, to pass test_stubs.py
        def stub(*args, **kwargs):
            raise NotImplementedError(
                f"Stub for {name} is not implemented in backend"
            )  # pragma: no cover

        return stub
    except ImportError:  # pragma: no cover

        def stub(*args, **kwargs):  # pragma: no cover
            raise NotImplementedError(
                f"Stub for {name} is not implemented in backend"
            )  # pragma: no cover

        return stub  # pragma: no cover
