"""Tests for zero_jax.interpreters.pxla."""

from __future__ import annotations

import zero_jax.interpreters.pxla as mod


def test_ArrayMapping() -> None:
    """Test ArrayMapping."""
    obj = mod.ArrayMapping({"x": 0}, id=1, name="test_ArrayMapping", value="test_value")
    assert obj["x"] == 0
    assert obj.id == 1
    assert obj.name == "test_ArrayMapping"
    assert obj.value == "test_value"


def test_Chunked() -> None:
    """Test Chunked."""
    obj = mod.Chunked(id=1, name="test_Chunked", value="test_value")
    assert obj.id == 1
    assert obj.name == "test_Chunked"
    assert obj.value == "test_value"


def test_Index() -> None:
    """Test Index."""
    assert mod.Index(0) == 0


def test_MapTracer() -> None:
    """Test MapTracer."""
    obj = mod.MapTracer(id=1, name="test_MapTracer", value="test_value")
    assert obj.id == 1
    assert obj.name == "test_MapTracer"
    assert obj.value == "test_value"


def test_MeshAxisName() -> None:
    """Test MeshAxisName."""
    assert mod.MeshAxisName("batch") == "batch"


def test_MeshComputation() -> None:
    """Test MeshComputation."""
    obj = mod.MeshComputation(id=1, name="test_MeshComputation", value="test_value")
    assert obj.id == 1
    assert obj.name == "test_MeshComputation"
    assert obj.value == "test_value"


def test_MeshExecutable() -> None:
    """Test MeshExecutable."""
    obj = mod.MeshExecutable(id=1, name="test_MeshExecutable", value="test_value")
    assert obj.id == 1
    assert obj.name == "test_MeshExecutable"
    assert obj.value == "test_value"


def test_NoSharding() -> None:
    """Test NoSharding."""
    obj = mod.NoSharding(id=1, name="test_NoSharding", value="test_value")
    assert obj.id == 1
    assert obj.name == "test_NoSharding"
    assert obj.value == "test_value"


def test_PmapExecutable() -> None:
    """Test PmapExecutable."""
    obj = mod.PmapExecutable(id=1, name="test_PmapExecutable", value="test_value")
    assert obj.id == 1
    assert obj.name == "test_PmapExecutable"
    assert obj.value == "test_value"


def test_Replicated() -> None:
    """Test Replicated."""
    obj = mod.Replicated(id=1, name="test_Replicated", value="test_value")
    assert obj.id == 1
    assert obj.name == "test_Replicated"
    assert obj.value == "test_value"


def test_ShardedAxis() -> None:
    """Test ShardedAxis."""
    obj = mod.ShardedAxis(id=1, name="test_ShardedAxis", value="test_value")
    assert obj.id == 1
    assert obj.name == "test_ShardedAxis"
    assert obj.value == "test_value"


def test_ShardingSpec() -> None:
    """Test ShardingSpec."""
    obj = mod.ShardingSpec(id=1, name="test_ShardingSpec", value="test_value")
    assert obj.id == 1
    assert obj.name == "test_ShardingSpec"
    assert obj.value == "test_value"


def test_Unstacked() -> None:
    """Test Unstacked."""
    obj = mod.Unstacked(id=1, name="test_Unstacked", value="test_value")
    assert obj.id == 1
    assert obj.name == "test_Unstacked"
    assert obj.value == "test_value"


def test_are_op_shardings_equal() -> None:
    """Test are_op_shardings_equal."""
    assert mod.are_op_shardings_equal("replicated", "replicated") is True
    assert mod.are_op_shardings_equal("replicated", "tiled") is False


def test_array_mapping_to_axis_resources() -> None:
    """Test array_mapping_to_axis_resources."""
    assert mod.array_mapping_to_axis_resources(mod.ArrayMapping({"x": 0, "y": 1})) == ("x", "y")
    assert mod.array_mapping_to_axis_resources(None) == ()


def test_global_aval_to_result_handler() -> None:
    """Test global_aval_to_result_handler."""
    handler = mod.global_aval_to_result_handler()
    assert handler(42) == 42


def test_global_avals_to_results_handler() -> None:
    """Test global_avals_to_results_handler."""
    handler = mod.global_avals_to_results_handler()
    assert handler([1, 2]) == [1, 2]


def test_is_op_sharding_replicated() -> None:
    """Test is_op_sharding_replicated."""
    assert mod.is_op_sharding_replicated(None) is True
    assert mod.is_op_sharding_replicated("tiled") is False


def test_op_sharding_to_indices() -> None:
    """Test op_sharding_to_indices."""
    indices = mod.op_sharding_to_indices("replicated", (4, 2), 2)
    assert len(indices) == 2
    assert indices[0] == (slice(0, 4), slice(0, 2))


def test_parallel_callable() -> None:
    """Test parallel_callable."""
    fn = lambda x: x * 2
    assert mod.parallel_callable(fn)(3) == 6
    default_fn = mod.parallel_callable()
    assert default_fn(1, 2) == (1, 2)


def test_shard_args() -> None:
    """Test shard_args."""
    assert mod.shard_args((), [1, 2]) == (1, 2)
    assert mod.shard_args((), 5) == (5,)


def test_spec_to_indices() -> None:
    """Test spec_to_indices."""
    assert mod.spec_to_indices((3, 4)) == ((slice(0, 3), slice(0, 4)),)
