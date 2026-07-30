"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.scipy.stats.binom as mod


def test_logpmf() -> None:
    """Test logpmf."""
    with patch("zero_jax._compiler_proxy_ops.logpmf", create=True) as mock_op:
        mod.logpmf()
        mock_op.assert_called_once_with()


def test_pmf() -> None:
    """Test pmf."""
    with patch("zero_jax._compiler_proxy_ops.pmf", create=True) as mock_op:
        mod.pmf()
        mock_op.assert_called_once_with()
