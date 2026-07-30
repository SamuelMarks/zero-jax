"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.scipy.stats.dirichlet as mod


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
