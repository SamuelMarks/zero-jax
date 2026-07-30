"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.scipy.stats as mod


def test_gaussian_kde() -> None:
    """Test gaussian_kde."""
    obj = mod.gaussian_kde()
    assert obj is not None


def test_mode() -> None:
    """Test mode."""
    with patch("zero_jax._compiler_proxy_ops.mode", create=True) as mock_op:
        mod.mode()
        mock_op.assert_called_once_with()


def test_rankdata() -> None:
    """Test rankdata."""
    with patch("zero_jax._compiler_proxy_ops.rankdata", create=True) as mock_op:
        mod.rankdata()
        mock_op.assert_called_once_with()


def test_sem() -> None:
    """Test sem."""
    with patch("zero_jax._compiler_proxy_ops.sem", create=True) as mock_op:
        mod.sem()
        mock_op.assert_called_once_with()
