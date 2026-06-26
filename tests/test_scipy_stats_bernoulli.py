"""Tests for zero_jax module."""

import pytest
import zero_jax.scipy.stats.bernoulli as mod


def test_cdf() -> None:
    """Test cdf."""
    with pytest.raises(NotImplementedError):
        mod.cdf()


def test_logpmf() -> None:
    """Test logpmf."""
    with pytest.raises(NotImplementedError):
        mod.logpmf()


def test_pmf() -> None:
    """Test pmf."""
    with pytest.raises(NotImplementedError):
        mod.pmf()


def test_ppf() -> None:
    """Test ppf."""
    with pytest.raises(NotImplementedError):
        mod.ppf()
