"""Tests for zero_jax module."""

import pytest
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
    with pytest.raises(NotImplementedError):
        mod.get_metadata()


def test_get_tpu_env_value() -> None:
    """Test get_tpu_env_value."""
    with pytest.raises(NotImplementedError):
        mod.get_tpu_env_value()


def test_has_megascale_address() -> None:
    """Test has_megascale_address."""
    with pytest.raises(NotImplementedError):
        mod.has_megascale_address()
