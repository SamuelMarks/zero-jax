"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
import zero_jax.custom_derivatives as mod


def test_CustomVJPPrimal() -> None:
    """Test CustomVJPPrimal."""
    obj = mod.CustomVJPPrimal()
    assert obj is not None


def test_SymbolicZero() -> None:
    """Test SymbolicZero."""
    obj = mod.SymbolicZero()
    assert obj is not None


def test_closure_convert() -> None:
    """Test closure_convert."""
    with patch("ml_switcheroo_compiler.ops.closure_convert") as mock_op:
        mod.closure_convert()
        mock_op.assert_called_once_with()


def test_custom_gradient() -> None:
    """Test custom_gradient."""
    with patch("ml_switcheroo_compiler.ops.custom_gradient") as mock_op:
        mod.custom_gradient()
        mock_op.assert_called_once_with()


def test_custom_jvp() -> None:
    """Test custom_jvp."""
    obj = mod.custom_jvp()
    assert obj is not None


def test_custom_vjp() -> None:
    """Test custom_vjp."""
    obj = mod.custom_vjp()
    assert obj is not None


def test_custom_vjp_primal_tree_values() -> None:
    """Test custom_vjp_primal_tree_values."""
    with patch("ml_switcheroo_compiler.ops.custom_vjp_primal_tree_values") as mock_op:
        mod.custom_vjp_primal_tree_values()
        mock_op.assert_called_once_with()


def test_linear_call() -> None:
    """Test linear_call."""
    with patch("ml_switcheroo_compiler.ops.linear_call") as mock_op:
        mod.linear_call()
        mock_op.assert_called_once_with()
