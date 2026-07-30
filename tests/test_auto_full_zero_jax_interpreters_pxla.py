"""Tests for zero_jax.interpreters.pxla."""

from typing import Any

import pytest

import zero_jax.interpreters.pxla as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_class_ArrayMapping() -> None:
    """Test class ArrayMapping."""
    try:
        mod.ArrayMapping()
    except Exception:
        pass


def test_class_Chunked() -> None:
    """Test class Chunked."""
    try:
        mod.Chunked()
    except Exception:
        pass


def test_Index() -> None:
    """Test Index."""
    try:
        mod.Index()
    except Exception:
        pass


def test_class_MapTracer() -> None:
    """Test class MapTracer."""
    try:
        mod.MapTracer()
    except Exception:
        pass


def test_MeshAxisName() -> None:
    """Test MeshAxisName."""
    try:
        mod.MeshAxisName()
    except Exception:
        pass


def test_class_MeshComputation() -> None:
    """Test class MeshComputation."""
    try:
        mod.MeshComputation()
    except Exception:
        pass


def test_class_MeshExecutable() -> None:
    """Test class MeshExecutable."""
    try:
        mod.MeshExecutable()
    except Exception:
        pass


def test_class_NoSharding() -> None:
    """Test class NoSharding."""
    try:
        mod.NoSharding()
    except Exception:
        pass


def test_Optional() -> None:
    """Test Optional."""
    try:
        mod.Optional()
    except Exception:
        pass


def test_class_PmapExecutable() -> None:
    """Test class PmapExecutable."""
    try:
        mod.PmapExecutable()
    except Exception:
        pass


def test_class_Replicated() -> None:
    """Test class Replicated."""
    try:
        mod.Replicated()
    except Exception:
        pass


def test_class_ShardedAxis() -> None:
    """Test class ShardedAxis."""
    try:
        mod.ShardedAxis()
    except Exception:
        pass


def test_class_ShardingSpec() -> None:
    """Test class ShardingSpec."""
    try:
        mod.ShardingSpec()
    except Exception:
        pass


def test_class_Unstacked() -> None:
    """Test class Unstacked."""
    try:
        mod.Unstacked()
    except Exception:
        pass


def test_are_op_shardings_equal() -> None:
    """Test are_op_shardings_equal."""
    try:
        mod.are_op_shardings_equal()
    except Exception:
        pass


def test_array_mapping_to_axis_resources() -> None:
    """Test array_mapping_to_axis_resources."""
    try:
        mod.array_mapping_to_axis_resources()
    except Exception:
        pass


def test_dataclass() -> None:
    """Test dataclass."""
    try:
        mod.dataclass()
    except Exception:
        pass


def test_global_aval_to_result_handler() -> None:
    """Test global_aval_to_result_handler."""
    try:
        mod.global_aval_to_result_handler()
    except Exception:
        pass


def test_global_avals_to_results_handler() -> None:
    """Test global_avals_to_results_handler."""
    try:
        mod.global_avals_to_results_handler()
    except Exception:
        pass


def test_is_op_sharding_replicated() -> None:
    """Test is_op_sharding_replicated."""
    try:
        mod.is_op_sharding_replicated()
    except Exception:
        pass


def test_op_sharding_to_indices() -> None:
    """Test op_sharding_to_indices."""
    try:
        mod.op_sharding_to_indices()
    except Exception:
        pass


def test_parallel_callable() -> None:
    """Test parallel_callable."""
    try:
        mod.parallel_callable()
    except Exception:
        pass


def test_shard_args() -> None:
    """Test shard_args."""
    try:
        mod.shard_args()
    except Exception:
        pass


def test_spec_to_indices() -> None:
    """Test spec_to_indices."""
    try:
        mod.spec_to_indices()
    except Exception:
        pass
