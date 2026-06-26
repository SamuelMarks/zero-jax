"""Tests for zero_jax module."""

import pytest
import zero_jax.interpreters.xla.xb.distributed.clusters.ompi_cluster as mod


def test_OmpiCluster() -> None:
    """Test OmpiCluster."""
    obj = mod.OmpiCluster()
    assert obj is not None
