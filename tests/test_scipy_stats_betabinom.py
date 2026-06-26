"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.scipy.stats.betabinom as mod


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
