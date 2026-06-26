"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.scipy.stats.bernoulli as mod


def test_cdf() -> None:
    """Test cdf."""
    with patch("ml_switcheroo_compiler.ops.cdf") as mock_op:
        mod.cdf()
        mock_op.assert_called_once_with()


def test_logpmf() -> None:
    """Test logpmf."""
    with patch("ml_switcheroo_compiler.ops.logpmf") as mock_op:
        mod.logpmf()
        mock_op.assert_called_once_with()


def test_pmf() -> None:
    """Test pmf."""
    with patch("ml_switcheroo_compiler.ops.pmf") as mock_op:
        mod.pmf()
        mock_op.assert_called_once_with()


def test_ppf() -> None:
    """Test ppf."""
    with patch("ml_switcheroo_compiler.ops.ppf") as mock_op:
        mod.ppf()
        mock_op.assert_called_once_with()
