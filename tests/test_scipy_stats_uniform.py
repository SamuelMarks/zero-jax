"""Tests for zero_jax module."""

import pytest
import zero_jax.scipy.stats.uniform as mod


def test_cdf() -> None:
    """Test cdf."""
    with pytest.raises(NotImplementedError):
        mod.cdf()


def test_logpdf() -> None:
    """Test logpdf."""
    with pytest.raises(NotImplementedError):
        mod.logpdf()


def test_pdf() -> None:
    """Test pdf."""
    with pytest.raises(NotImplementedError):
        mod.pdf()


def test_ppf() -> None:
    """Test ppf."""
    with pytest.raises(NotImplementedError):
        mod.ppf()
