"""Tests for zero_jax module."""

import pytest
import zero_jax.scipy.stats.betabinom as mod


def test_logpmf() -> None:
    """Test logpmf."""
    with pytest.raises(NotImplementedError):
        mod.logpmf()


def test_pmf() -> None:
    """Test pmf."""
    with pytest.raises(NotImplementedError):
        mod.pmf()
