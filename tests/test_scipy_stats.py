"""Tests for zero_jax module."""

import pytest
import zero_jax.scipy.stats as mod


def test_gaussian_kde() -> None:
    """Test gaussian_kde."""
    obj = mod.gaussian_kde()
    assert obj is not None


def test_mode() -> None:
    """Test mode."""
    with pytest.raises(NotImplementedError):
        mod.mode()


def test_rankdata() -> None:
    """Test rankdata."""
    with pytest.raises(NotImplementedError):
        mod.rankdata()


def test_sem() -> None:
    """Test sem."""
    with pytest.raises(NotImplementedError):
        mod.sem()
