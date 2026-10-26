"""Tests for zero_jax.scipy.sparse."""

from __future__ import annotations

import zero_jax.scipy.sparse as mod


def test_module_exists() -> None:
    """Test scipy.sparse exposes linalg and sparse classes."""
    assert "linalg" in mod.__all__
    assert isinstance(mod.SparseEye(), mod.SparseEye)
