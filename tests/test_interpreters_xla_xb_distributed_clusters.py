"""Tests for zero_jax module."""

import pytest
import zero_jax.interpreters.xla.xb.distributed.clusters as mod


def test_ClusterEnv() -> None:
    """Test ClusterEnv."""
    obj = mod.ClusterEnv()
    assert obj is not None


def test_GceTpuCluster() -> None:
    """Test GceTpuCluster."""
    obj = mod.GceTpuCluster()
    assert obj is not None


def test_GkeTpuCluster() -> None:
    """Test GkeTpuCluster."""
    obj = mod.GkeTpuCluster()
    assert obj is not None


def test_OmpiCluster() -> None:
    """Test OmpiCluster."""
    obj = mod.OmpiCluster()
    assert obj is not None


def test_SlurmCluster() -> None:
    """Test SlurmCluster."""
    obj = mod.SlurmCluster()
    assert obj is not None
