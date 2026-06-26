"""Tests for zero_jax module."""

import pytest
import zero_jax.interpreters.xla.xb.distributed.clusters.slurm_cluster as mod


def test_SlurmCluster() -> None:
    """Test SlurmCluster."""
    obj = mod.SlurmCluster()
    assert obj is not None
