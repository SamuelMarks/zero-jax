"""Tests for zero_jax module."""

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
    with pytest.raises(NotImplementedError):
        mod.closure_convert()


def test_custom_gradient() -> None:
    """Test custom_gradient."""
    with pytest.raises(NotImplementedError):
        mod.custom_gradient()


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
    with pytest.raises(NotImplementedError):
        mod.custom_vjp_primal_tree_values()


def test_linear_call() -> None:
    """Test linear_call."""
    with pytest.raises(NotImplementedError):
        mod.linear_call()
