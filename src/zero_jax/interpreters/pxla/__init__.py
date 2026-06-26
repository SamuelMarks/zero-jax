"""Frontend API routing for jax.interpreters.pxla."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


class ArrayMapping:
    """Dictionary that remembers insertion order"""

    pass


class Chunked:
    """Mock implementation for Chunked."""

    pass


def Index(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for Index."""
    return getattr(_ops, "Index")(*args, **kwargs)


class MapTracer:
    """Mock implementation for MapTracer."""

    pass


def MeshAxisName(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating an unconstrained type."""
    return getattr(_ops, "MeshAxisName")(*args, **kwargs)


class MeshComputation:
    """Mock implementation for MeshComputation."""

    pass


class MeshExecutable:
    """Mock implementation for MeshExecutable."""

    pass


class NoSharding:
    """Mock implementation for NoSharding."""

    pass


class PmapExecutable:
    """Mock implementation for PmapExecutable."""

    pass


class Replicated:
    """Mock implementation for Replicated."""

    pass


class ShardedAxis:
    """Mock implementation for ShardedAxis."""

    pass


class ShardingSpec:
    """Mock implementation for ShardingSpec."""

    pass


class Unstacked:
    """Mock implementation for Unstacked."""

    pass


def are_op_shardings_equal(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for are_op_shardings_equal."""
    return getattr(_ops, "are_op_shardings_equal")(*args, **kwargs)


def array_mapping_to_axis_resources(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for array_mapping_to_axis_resources."""
    return getattr(_ops, "array_mapping_to_axis_resources")(*args, **kwargs)


def global_aval_to_result_handler(*args: Any, **kwargs: Any) -> Any:
    """Returns a function for handling the raw buffers of a single output aval."""
    return getattr(_ops, "global_aval_to_result_handler")(*args, **kwargs)


def global_avals_to_results_handler(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for global_avals_to_results_handler."""
    return getattr(_ops, "global_avals_to_results_handler")(*args, **kwargs)


global_result_handlers: Any = None


def is_op_sharding_replicated(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for is_op_sharding_replicated."""
    return getattr(_ops, "is_op_sharding_replicated")(*args, **kwargs)


def op_sharding_to_indices(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for op_sharding_to_indices."""
    return getattr(_ops, "op_sharding_to_indices")(*args, **kwargs)


def parallel_callable(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for parallel_callable."""
    return getattr(_ops, "parallel_callable")(*args, **kwargs)


def shard_args(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for shard_args."""
    return getattr(_ops, "shard_args")(*args, **kwargs)


def spec_to_indices(*args: Any, **kwargs: Any) -> Any:
    """Returns numpy-style indices corresponding to a sharding spec."""
    return getattr(_ops, "spec_to_indices")(*args, **kwargs)


thread_resources: Any = None

xla_pmap_p: Any = None
