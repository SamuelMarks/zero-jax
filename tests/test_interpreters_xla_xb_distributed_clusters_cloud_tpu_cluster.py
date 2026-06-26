"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.interpreters.xla.xb.distributed.clusters.cloud_tpu_cluster as mod


def test_BaseTpuCluster() -> None:
    """Test BaseTpuCluster."""
    obj = mod.BaseTpuCluster()
    assert obj is not None


def test_GceTpuCluster() -> None:
    """Test GceTpuCluster."""
    obj = mod.GceTpuCluster()
    assert obj is not None


def test_GkeTpuCluster() -> None:
    """Test GkeTpuCluster."""
    obj = mod.GkeTpuCluster()
    assert obj is not None


def test_get_metadata() -> None:
    """Test get_metadata."""
    with patch("ml_switcheroo_compiler.ops.get_metadata") as mock_op:
        mod.get_metadata()
        mock_op.assert_called_once_with()


def test_get_tpu_env_value() -> None:
    """Test get_tpu_env_value."""
    with patch("ml_switcheroo_compiler.ops.get_tpu_env_value") as mock_op:
        mod.get_tpu_env_value()
        mock_op.assert_called_once_with()


def test_has_megascale_address() -> None:
    """Test has_megascale_address."""
    with patch("ml_switcheroo_compiler.ops.has_megascale_address") as mock_op:
        mod.has_megascale_address()
        mock_op.assert_called_once_with()
