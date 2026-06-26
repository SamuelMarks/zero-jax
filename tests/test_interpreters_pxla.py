"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
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
    with patch("ml_switcheroo_compiler.ops.Index") as mock_op:
        mod.Index()
        mock_op.assert_called_once_with()


def test_MapTracer() -> None:
    """Test MapTracer."""
    obj = mod.MapTracer()
    assert obj is not None


def test_MeshAxisName() -> None:
    """Test MeshAxisName."""
    with patch("ml_switcheroo_compiler.ops.MeshAxisName") as mock_op:
        mod.MeshAxisName()
        mock_op.assert_called_once_with()


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
    with patch("ml_switcheroo_compiler.ops.are_op_shardings_equal") as mock_op:
        mod.are_op_shardings_equal()
        mock_op.assert_called_once_with()


def test_array_mapping_to_axis_resources() -> None:
    """Test array_mapping_to_axis_resources."""
    with patch("ml_switcheroo_compiler.ops.array_mapping_to_axis_resources") as mock_op:
        mod.array_mapping_to_axis_resources()
        mock_op.assert_called_once_with()


def test_global_aval_to_result_handler() -> None:
    """Test global_aval_to_result_handler."""
    with patch("ml_switcheroo_compiler.ops.global_aval_to_result_handler") as mock_op:
        mod.global_aval_to_result_handler()
        mock_op.assert_called_once_with()


def test_global_avals_to_results_handler() -> None:
    """Test global_avals_to_results_handler."""
    with patch("ml_switcheroo_compiler.ops.global_avals_to_results_handler") as mock_op:
        mod.global_avals_to_results_handler()
        mock_op.assert_called_once_with()


def test_is_op_sharding_replicated() -> None:
    """Test is_op_sharding_replicated."""
    with patch("ml_switcheroo_compiler.ops.is_op_sharding_replicated") as mock_op:
        mod.is_op_sharding_replicated()
        mock_op.assert_called_once_with()


def test_op_sharding_to_indices() -> None:
    """Test op_sharding_to_indices."""
    with patch("ml_switcheroo_compiler.ops.op_sharding_to_indices") as mock_op:
        mod.op_sharding_to_indices()
        mock_op.assert_called_once_with()


def test_parallel_callable() -> None:
    """Test parallel_callable."""
    with patch("ml_switcheroo_compiler.ops.parallel_callable") as mock_op:
        mod.parallel_callable()
        mock_op.assert_called_once_with()


def test_shard_args() -> None:
    """Test shard_args."""
    with patch("ml_switcheroo_compiler.ops.shard_args") as mock_op:
        mod.shard_args()
        mock_op.assert_called_once_with()


def test_spec_to_indices() -> None:
    """Test spec_to_indices."""
    with patch("ml_switcheroo_compiler.ops.spec_to_indices") as mock_op:
        mod.spec_to_indices()
        mock_op.assert_called_once_with()
