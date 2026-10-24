"""Tests for tests/test_numpy_added.py."""

from __future__ import annotations

import zero_jax.numpy as jnp


def test_complex_warning() -> None:
    """Test ComplexWarning."""
    assert jnp.ComplexWarning is not None
