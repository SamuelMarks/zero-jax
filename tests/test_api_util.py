"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.api_util as mod


def test_argnums_partial() -> None:
    """Test argnums_partial."""
    with patch("ml_switcheroo_compiler.ops.argnums_partial") as mock_op:
        mod.argnums_partial()
        mock_op.assert_called_once_with()


def test_donation_vector() -> None:
    """Test donation_vector."""
    with patch("ml_switcheroo_compiler.ops.donation_vector") as mock_op:
        mod.donation_vector()
        mock_op.assert_called_once_with()


def test_flatten_axes() -> None:
    """Test flatten_axes."""
    with patch("ml_switcheroo_compiler.ops.flatten_axes") as mock_op:
        mod.flatten_axes()
        mock_op.assert_called_once_with()


def test_flatten_fun() -> None:
    """Test flatten_fun."""
    with patch("ml_switcheroo_compiler.ops.flatten_fun") as mock_op:
        mod.flatten_fun()
        mock_op.assert_called_once_with()


def test_flatten_fun_nokwargs() -> None:
    """Test flatten_fun_nokwargs."""
    with patch("ml_switcheroo_compiler.ops.flatten_fun_nokwargs") as mock_op:
        mod.flatten_fun_nokwargs()
        mock_op.assert_called_once_with()


def test_rebase_donate_argnums() -> None:
    """Test rebase_donate_argnums."""
    with patch("ml_switcheroo_compiler.ops.rebase_donate_argnums") as mock_op:
        mod.rebase_donate_argnums()
        mock_op.assert_called_once_with()


def test_safe_map() -> None:
    """Test safe_map."""
    with patch("ml_switcheroo_compiler.ops.safe_map") as mock_op:
        mod.safe_map()
        mock_op.assert_called_once_with()


def test_shaped_abstractify() -> None:
    """Test shaped_abstractify."""
    with patch("ml_switcheroo_compiler.ops.shaped_abstractify") as mock_op:
        mod.shaped_abstractify()
        mock_op.assert_called_once_with()
