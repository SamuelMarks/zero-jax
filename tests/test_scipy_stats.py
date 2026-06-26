"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.scipy.stats as mod


def test_gaussian_kde() -> None:
    """Test gaussian_kde."""
    obj = mod.gaussian_kde()
    assert obj is not None


def test_mode() -> None:
    """Test mode."""
    with patch("ml_switcheroo_compiler.ops.mode") as mock_op:
        mod.mode()
        mock_op.assert_called_once_with()


def test_rankdata() -> None:
    """Test rankdata."""
    with patch("ml_switcheroo_compiler.ops.rankdata") as mock_op:
        mod.rankdata()
        mock_op.assert_called_once_with()


def test_sem() -> None:
    """Test sem."""
    with patch("ml_switcheroo_compiler.ops.sem") as mock_op:
        mod.sem()
        mock_op.assert_called_once_with()
