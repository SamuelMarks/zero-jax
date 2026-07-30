"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

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
    with patch("zero_jax._compiler_proxy_ops.closure_convert", create=True) as mock_op:
        mod.closure_convert()
        mock_op.assert_called_once_with()


def test_custom_gradient() -> None:
    """Test custom_gradient."""
    with patch("zero_jax._compiler_proxy_ops.custom_gradient", create=True) as mock_op:
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
    with patch(
        "zero_jax._compiler_proxy_ops.custom_vjp_primal_tree_values", create=True
    ) as mock_op:
        mod.custom_vjp_primal_tree_values()
        mock_op.assert_called_once_with()


def test_linear_call() -> None:
    """Test linear_call."""
    with patch("zero_jax._compiler_proxy_ops.linear_call", create=True) as mock_op:
        mod.linear_call()
        mock_op.assert_called_once_with()
