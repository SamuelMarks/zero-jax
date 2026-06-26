"""Tests for zero_jax module."""

import pytest
import zero_jax.tree as mod


def test_all() -> None:
    """Test all."""
    with pytest.raises(NotImplementedError):
        mod.all()


def test_flatten() -> None:
    """Test flatten."""
    with pytest.raises(NotImplementedError):
        mod.flatten()


def test_leaves() -> None:
    """Test leaves."""
    with pytest.raises(NotImplementedError):
        mod.leaves()


def test_map() -> None:
    """Test map."""
    with pytest.raises(NotImplementedError):
        mod.map()


def test_reduce() -> None:
    """Test reduce."""
    with pytest.raises(NotImplementedError):
        mod.reduce()


def test_structure() -> None:
    """Test structure."""
    with pytest.raises(NotImplementedError):
        mod.structure()


def test_transpose() -> None:
    """Test transpose."""
    with pytest.raises(NotImplementedError):
        mod.transpose()


def test_unflatten() -> None:
    """Test unflatten."""
    with pytest.raises(NotImplementedError):
        mod.unflatten()
