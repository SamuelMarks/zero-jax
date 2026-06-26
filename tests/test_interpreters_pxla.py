"""Tests for zero_jax module."""

import pytest
import zero_jax.interpreters.pxla as mod


def test_ArrayMapping() -> None:
    """Test ArrayMapping."""
    obj = mod.ArrayMapping()
    assert obj is not None


def test_Chunked() -> None:
    """Test Chunked."""
    obj = mod.Chunked()
    assert obj is not None


def test_Index() -> None:
    """Test Index."""
    with pytest.raises(NotImplementedError):
        mod.Index()


def test_MapTracer() -> None:
    """Test MapTracer."""
    obj = mod.MapTracer()
    assert obj is not None


def test_MeshAxisName() -> None:
    """Test MeshAxisName."""
    with pytest.raises(NotImplementedError):
        mod.MeshAxisName()


def test_MeshComputation() -> None:
    """Test MeshComputation."""
    obj = mod.MeshComputation()
    assert obj is not None


def test_MeshExecutable() -> None:
    """Test MeshExecutable."""
    obj = mod.MeshExecutable()
    assert obj is not None


def test_NoSharding() -> None:
    """Test NoSharding."""
    obj = mod.NoSharding()
    assert obj is not None


def test_PmapExecutable() -> None:
    """Test PmapExecutable."""
    obj = mod.PmapExecutable()
    assert obj is not None


def test_Replicated() -> None:
    """Test Replicated."""
    obj = mod.Replicated()
    assert obj is not None


def test_ShardedAxis() -> None:
    """Test ShardedAxis."""
    obj = mod.ShardedAxis()
    assert obj is not None


def test_ShardingSpec() -> None:
    """Test ShardingSpec."""
    obj = mod.ShardingSpec()
    assert obj is not None


def test_Unstacked() -> None:
    """Test Unstacked."""
    obj = mod.Unstacked()
    assert obj is not None


def test_are_op_shardings_equal() -> None:
    """Test are_op_shardings_equal."""
    with pytest.raises(NotImplementedError):
        mod.are_op_shardings_equal()


def test_array_mapping_to_axis_resources() -> None:
    """Test array_mapping_to_axis_resources."""
    with pytest.raises(NotImplementedError):
        mod.array_mapping_to_axis_resources()


def test_global_aval_to_result_handler() -> None:
    """Test global_aval_to_result_handler."""
    with pytest.raises(NotImplementedError):
        mod.global_aval_to_result_handler()


def test_global_avals_to_results_handler() -> None:
    """Test global_avals_to_results_handler."""
    with pytest.raises(NotImplementedError):
        mod.global_avals_to_results_handler()


def test_is_op_sharding_replicated() -> None:
    """Test is_op_sharding_replicated."""
    with pytest.raises(NotImplementedError):
        mod.is_op_sharding_replicated()


def test_op_sharding_to_indices() -> None:
    """Test op_sharding_to_indices."""
    with pytest.raises(NotImplementedError):
        mod.op_sharding_to_indices()


def test_parallel_callable() -> None:
    """Test parallel_callable."""
    with pytest.raises(NotImplementedError):
        mod.parallel_callable()


def test_shard_args() -> None:
    """Test shard_args."""
    with pytest.raises(NotImplementedError):
        mod.shard_args()


def test_spec_to_indices() -> None:
    """Test spec_to_indices."""
    with pytest.raises(NotImplementedError):
        mod.spec_to_indices()
