"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.scipy.stats.uniform as mod


def test_cdf() -> None:
    """Test cdf."""
    with patch("zero_jax._compiler_proxy_ops.cdf", create=True) as mock_op:
        mod.cdf()
        mock_op.assert_called_once_with()


def test_logpdf() -> None:
    """Test logpdf."""
    with patch("zero_jax._compiler_proxy_ops.logpdf", create=True) as mock_op:
        mod.logpdf()
        mock_op.assert_called_once_with()


def test_pdf() -> None:
    """Test pdf."""
    with patch("zero_jax._compiler_proxy_ops.pdf", create=True) as mock_op:
        mod.pdf()
        mock_op.assert_called_once_with()


def test_ppf() -> None:
    """Test ppf."""
    with patch("zero_jax._compiler_proxy_ops.ppf", create=True) as mock_op:
        mod.ppf()
        mock_op.assert_called_once_with()
