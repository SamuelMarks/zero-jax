"""Tests for zero_jax module."""

import pytest
import zero_jax.interpreters.xla.xb.distributed.clusters.cluster as mod


def test_ClusterEnv() -> None:
    """Test ClusterEnv."""
    obj = mod.ClusterEnv()
    assert obj is not None
