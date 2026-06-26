"""Tests for zero_jax module."""

import pytest
import zero_jax.scipy.stats.truncnorm as mod


def test_cdf() -> None:
    """Test cdf."""
    with pytest.raises(NotImplementedError):
        mod.cdf()


def test_logcdf() -> None:
    """Test logcdf."""
    with pytest.raises(NotImplementedError):
        mod.logcdf()


def test_logpdf() -> None:
    """Test logpdf."""
    with pytest.raises(NotImplementedError):
        mod.logpdf()


def test_logsf() -> None:
    """Test logsf."""
    with pytest.raises(NotImplementedError):
        mod.logsf()


def test_pdf() -> None:
    """Test pdf."""
    with pytest.raises(NotImplementedError):
        mod.pdf()


def test_sf() -> None:
    """Test sf."""
    with pytest.raises(NotImplementedError):
        mod.sf()
