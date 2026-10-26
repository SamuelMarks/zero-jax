"""Tests for zero_jax.lib.xla_bridge."""

from __future__ import annotations

import zero_jax.lib.xla_bridge as mod


def test_default_backend() -> None:
    """Test default_backend."""
    assert mod.default_backend() == "cpu"


def test_get_backend() -> None:
    """Test get_backend."""
    backend = mod.get_backend("cpu")
    assert backend.platform == "cpu"


def test_get_compile_options() -> None:
    """Test get_compile_options."""
    opts = mod.get_compile_options(num_replicas=2, num_partitions=1)
    assert opts["num_replicas"] == 2
    assert opts["num_partitions"] == 1
