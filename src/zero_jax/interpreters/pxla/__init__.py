"""Frontend API routing for jax.interpreters.pxla."""

from typing import Any


class ArrayMapping:
    """Dictionary that remembers insertion order"""

    pass


class Chunked:
    """Mock implementation for Chunked."""

    pass


def Index(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for Index."""
    raise NotImplementedError("Index not yet implemented in zero-jax")


class MapTracer:
    """Mock implementation for MapTracer."""

    pass


def MeshAxisName(*args: Any, **kwargs: Any) -> Any:
    """Special type indicating an unconstrained type."""
    raise NotImplementedError("MeshAxisName not yet implemented in zero-jax")


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
    raise NotImplementedError("are_op_shardings_equal not yet implemented in zero-jax")


def array_mapping_to_axis_resources(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for array_mapping_to_axis_resources."""
    raise NotImplementedError(
        "array_mapping_to_axis_resources not yet implemented in zero-jax"
    )


def global_aval_to_result_handler(*args: Any, **kwargs: Any) -> Any:
    """Returns a function for handling the raw buffers of a single output aval."""
    raise NotImplementedError(
        "global_aval_to_result_handler not yet implemented in zero-jax"
    )


def global_avals_to_results_handler(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for global_avals_to_results_handler."""
    raise NotImplementedError(
        "global_avals_to_results_handler not yet implemented in zero-jax"
    )


global_result_handlers: Any = None


def is_op_sharding_replicated(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for is_op_sharding_replicated."""
    raise NotImplementedError(
        "is_op_sharding_replicated not yet implemented in zero-jax"
    )


def op_sharding_to_indices(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for op_sharding_to_indices."""
    raise NotImplementedError("op_sharding_to_indices not yet implemented in zero-jax")


def parallel_callable(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for parallel_callable."""
    raise NotImplementedError("parallel_callable not yet implemented in zero-jax")


def shard_args(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for shard_args."""
    raise NotImplementedError("shard_args not yet implemented in zero-jax")


def spec_to_indices(*args: Any, **kwargs: Any) -> Any:
    """Returns numpy-style indices corresponding to a sharding spec."""
    raise NotImplementedError("spec_to_indices not yet implemented in zero-jax")


thread_resources: Any = None

xla_pmap_p: Any = None
