"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.scipy.stats.dirichlet as mod


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
