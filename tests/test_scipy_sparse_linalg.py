"""Tests for zero_jax module."""

import pytest
import zero_jax.scipy.sparse.linalg as mod


def test_bicgstab() -> None:
    """Test bicgstab."""
    with pytest.raises(NotImplementedError):
        mod.bicgstab()


def test_cg() -> None:
    """Test cg."""
    with pytest.raises(NotImplementedError):
        mod.cg()


def test_gmres() -> None:
    """Test gmres."""
    with pytest.raises(NotImplementedError):
        mod.gmres()
