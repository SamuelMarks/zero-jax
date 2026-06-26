"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.scipy.stats.logistic as mod


def test_cdf() -> None:
    """Test cdf."""
    with patch("ml_switcheroo_compiler.ops.cdf") as mock_op:
        mod.cdf()
        mock_op.assert_called_once_with()


def test_isf() -> None:
    """Test isf."""
    with patch("ml_switcheroo_compiler.ops.isf") as mock_op:
        mod.isf()
        mock_op.assert_called_once_with()


def test_logpdf() -> None:
    """Test logpdf."""
    with patch("ml_switcheroo_compiler.ops.logpdf") as mock_op:
        mod.logpdf()
        mock_op.assert_called_once_with()


def test_pdf() -> None:
    """Test pdf."""
    with patch("ml_switcheroo_compiler.ops.pdf") as mock_op:
        mod.pdf()
        mock_op.assert_called_once_with()


def test_ppf() -> None:
    """Test ppf."""
    with patch("ml_switcheroo_compiler.ops.ppf") as mock_op:
        mod.ppf()
        mock_op.assert_called_once_with()


def test_sf() -> None:
    """Test sf."""
    with patch("ml_switcheroo_compiler.ops.sf") as mock_op:
        mod.sf()
        mock_op.assert_called_once_with()
