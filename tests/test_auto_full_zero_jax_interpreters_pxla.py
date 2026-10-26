"""Tests for zero_jax.interpreters.pxla."""

from __future__ import annotations

import zero_jax.interpreters.pxla as mod


def test_class_ArrayMapping() -> None:
    """Test class ArrayMapping."""
    assert isinstance(mod.ArrayMapping(), mod.ArrayMapping)


def test_class_Chunked() -> None:
    """Test class Chunked."""
    assert isinstance(mod.Chunked(), mod.Chunked)


def test_Index() -> None:
    """Test Index."""
    assert mod.Index(1) == 1


def test_class_MapTracer() -> None:
    """Test class MapTracer."""
    assert isinstance(mod.MapTracer(), mod.MapTracer)


def test_MeshAxisName() -> None:
    """Test MeshAxisName."""
    assert mod.MeshAxisName("x") == "x"


def test_class_MeshComputation() -> None:
    """Test class MeshComputation."""
    assert isinstance(mod.MeshComputation(), mod.MeshComputation)


def test_class_MeshExecutable() -> None:
    """Test class MeshExecutable."""
    assert isinstance(mod.MeshExecutable(), mod.MeshExecutable)


def test_class_NoSharding() -> None:
    """Test class NoSharding."""
    assert isinstance(mod.NoSharding(), mod.NoSharding)


def test_Optional() -> None:
    """Test Optional."""
    assert mod.Optional is not None


def test_class_PmapExecutable() -> None:
    """Test class PmapExecutable."""
    assert isinstance(mod.PmapExecutable(), mod.PmapExecutable)


def test_class_Replicated() -> None:
    """Test class Replicated."""
    assert isinstance(mod.Replicated(), mod.Replicated)


def test_class_ShardedAxis() -> None:
    """Test class ShardedAxis."""
    assert isinstance(mod.ShardedAxis(), mod.ShardedAxis)


def test_class_ShardingSpec() -> None:
    """Test class ShardingSpec."""
    assert isinstance(mod.ShardingSpec(), mod.ShardingSpec)


def test_class_Unstacked() -> None:
    """Test class Unstacked."""
    assert isinstance(mod.Unstacked(), mod.Unstacked)


def test_are_op_shardings_equal() -> None:
    """Test are_op_shardings_equal."""
    assert mod.are_op_shardings_equal("a", "a") is True


def test_array_mapping_to_axis_resources() -> None:
    """Test array_mapping_to_axis_resources."""
    assert mod.array_mapping_to_axis_resources({"x": 0}) == ("x",)


def test_dataclass() -> None:
    """Test dataclass."""
    assert callable(mod.dataclass)


def test_global_aval_to_result_handler() -> None:
    """Test global_aval_to_result_handler."""
    assert mod.global_aval_to_result_handler()(10) == 10


def test_global_avals_to_results_handler() -> None:
    """Test global_avals_to_results_handler."""
    assert mod.global_avals_to_results_handler()((10,)) == (10,)


def test_is_op_sharding_replicated() -> None:
    """Test is_op_sharding_replicated."""
    assert mod.is_op_sharding_replicated() is True


def test_op_sharding_to_indices() -> None:
    """Test op_sharding_to_indices."""
    assert mod.op_sharding_to_indices() == ((slice(0, 1),),)


def test_parallel_callable() -> None:
    """Test parallel_callable."""
    assert callable(mod.parallel_callable())


def test_shard_args() -> None:
    """Test shard_args."""
    assert mod.shard_args((), (1, 2)) == (1, 2)


def test_spec_to_indices() -> None:
    """Test spec_to_indices."""
    assert mod.spec_to_indices((2,)) == ((slice(0, 2),),)
