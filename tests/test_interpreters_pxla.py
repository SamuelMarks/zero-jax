"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.interpreters.pxla as mod


def test_ArrayMapping() -> None:
    """Test ArrayMapping."""
    obj = mod.ArrayMapping(id=1, name="test_ArrayMapping", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_ArrayMapping"
    assert obj.value == "test_value"


def test_Chunked() -> None:
    """Test Chunked."""
    obj = mod.Chunked(id=1, name="test_Chunked", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_Chunked"
    assert obj.value == "test_value"


def test_Index() -> None:
    """Test Index."""
    with patch("zero_jax._compiler_proxy_ops.Index", create=True) as mock_op:
        mod.Index()
        mock_op.assert_called_once_with()


def test_MapTracer() -> None:
    """Test MapTracer."""
    obj = mod.MapTracer(id=1, name="test_MapTracer", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_MapTracer"
    assert obj.value == "test_value"


def test_MeshAxisName() -> None:
    """Test MeshAxisName."""
    with patch("zero_jax._compiler_proxy_ops.MeshAxisName", create=True) as mock_op:
        mod.MeshAxisName()
        mock_op.assert_called_once_with()


def test_MeshComputation() -> None:
    """Test MeshComputation."""
    obj = mod.MeshComputation(id=1, name="test_MeshComputation", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_MeshComputation"
    assert obj.value == "test_value"


def test_MeshExecutable() -> None:
    """Test MeshExecutable."""
    obj = mod.MeshExecutable(id=1, name="test_MeshExecutable", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_MeshExecutable"
    assert obj.value == "test_value"


def test_NoSharding() -> None:
    """Test NoSharding."""
    obj = mod.NoSharding(id=1, name="test_NoSharding", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_NoSharding"
    assert obj.value == "test_value"


def test_PmapExecutable() -> None:
    """Test PmapExecutable."""
    obj = mod.PmapExecutable(id=1, name="test_PmapExecutable", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_PmapExecutable"
    assert obj.value == "test_value"


def test_Replicated() -> None:
    """Test Replicated."""
    obj = mod.Replicated(id=1, name="test_Replicated", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_Replicated"
    assert obj.value == "test_value"


def test_ShardedAxis() -> None:
    """Test ShardedAxis."""
    obj = mod.ShardedAxis(id=1, name="test_ShardedAxis", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_ShardedAxis"
    assert obj.value == "test_value"


def test_ShardingSpec() -> None:
    """Test ShardingSpec."""
    obj = mod.ShardingSpec(id=1, name="test_ShardingSpec", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_ShardingSpec"
    assert obj.value == "test_value"


def test_Unstacked() -> None:
    """Test Unstacked."""
    obj = mod.Unstacked(id=1, name="test_Unstacked", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_Unstacked"
    assert obj.value == "test_value"


def test_are_op_shardings_equal() -> None:
    """Test are_op_shardings_equal."""
    with patch(
        "zero_jax._compiler_proxy_ops.are_op_shardings_equal", create=True
    ) as mock_op:
        mod.are_op_shardings_equal()
        mock_op.assert_called_once_with()


def test_array_mapping_to_axis_resources() -> None:
    """Test array_mapping_to_axis_resources."""
    with patch(
        "zero_jax._compiler_proxy_ops.array_mapping_to_axis_resources", create=True
    ) as mock_op:
        mod.array_mapping_to_axis_resources()
        mock_op.assert_called_once_with()


def test_global_aval_to_result_handler() -> None:
    """Test global_aval_to_result_handler."""
    with patch(
        "zero_jax._compiler_proxy_ops.global_aval_to_result_handler", create=True
    ) as mock_op:
        mod.global_aval_to_result_handler()
        mock_op.assert_called_once_with()


def test_global_avals_to_results_handler() -> None:
    """Test global_avals_to_results_handler."""
    with patch(
        "zero_jax._compiler_proxy_ops.global_avals_to_results_handler", create=True
    ) as mock_op:
        mod.global_avals_to_results_handler()
        mock_op.assert_called_once_with()


def test_is_op_sharding_replicated() -> None:
    """Test is_op_sharding_replicated."""
    with patch(
        "zero_jax._compiler_proxy_ops.is_op_sharding_replicated", create=True
    ) as mock_op:
        mod.is_op_sharding_replicated()
        mock_op.assert_called_once_with()


def test_op_sharding_to_indices() -> None:
    """Test op_sharding_to_indices."""
    with patch(
        "zero_jax._compiler_proxy_ops.op_sharding_to_indices", create=True
    ) as mock_op:
        mod.op_sharding_to_indices()
        mock_op.assert_called_once_with()


def test_parallel_callable() -> None:
    """Test parallel_callable."""
    with patch(
        "zero_jax._compiler_proxy_ops.parallel_callable", create=True
    ) as mock_op:
        mod.parallel_callable()
        mock_op.assert_called_once_with()


def test_shard_args() -> None:
    """Test shard_args."""
    with patch("zero_jax._compiler_proxy_ops.shard_args", create=True) as mock_op:
        mod.shard_args()
        mock_op.assert_called_once_with()


def test_spec_to_indices() -> None:
    """Test spec_to_indices."""
    with patch("zero_jax._compiler_proxy_ops.spec_to_indices", create=True) as mock_op:
        mod.spec_to_indices()
        mock_op.assert_called_once_with()
