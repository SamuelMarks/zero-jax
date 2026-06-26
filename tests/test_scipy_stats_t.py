"""Tests for zero_jax module."""

import pytest
import zero_jax.scipy.stats.t as mod


def test_logpdf() -> None:
    """Test logpdf."""
    with pytest.raises(NotImplementedError):
        mod.logpdf()


def test_pdf() -> None:
    """Test pdf."""
    with pytest.raises(NotImplementedError):
        mod.pdf()
