"""Tests for zero_jax.ops."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.ops as mod
from tests.conftest import assert_allclose


def test_segment_max() -> None:
    """Test segment_max."""
    data = jnp.array([1.0, 5.0, 3.0])
    seg_ids = jnp.array([0, 0, 1], dtype="int32")
    assert_allclose(mod.segment_max(data, seg_ids, 2), [5.0, 3.0])


def test_segment_min() -> None:
    """Test segment_min."""
    data = jnp.array([1.0, 5.0, 3.0])
    seg_ids = jnp.array([0, 0, 1], dtype="int32")
    assert_allclose(mod.segment_min(data, seg_ids, 2), [1.0, 3.0])


def test_segment_prod() -> None:
    """Test segment_prod."""
    data = jnp.array([2.0, 3.0, 4.0])
    seg_ids = jnp.array([0, 0, 1], dtype="int32")
    assert_allclose(mod.segment_prod(data, seg_ids, 2), [6.0, 4.0])


def test_segment_sum() -> None:
    """Test segment_sum."""
    data = jnp.array([1.0, 2.0, 3.0])
    seg_ids = jnp.array([0, 0, 1], dtype="int32")
    assert_allclose(mod.segment_sum(data, seg_ids, 2), [3.0, 3.0])


def test_Transpose_exists() -> None:
    """Test Transpose exists in ml_switcheroo_compiler."""
    import zero_jax._compiler_proxy_ops as ops

    assert ops.Transpose is not None
    assert "Transpose" in ops.__all__
