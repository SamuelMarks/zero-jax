"""Tests for zero_jax.custom_derivatives."""

from __future__ import annotations

import pytest

import zero_jax.custom_derivatives as mod


def test_class_CustomVJPPrimal() -> None:
    """Test class CustomVJPPrimal."""
    obj = mod.CustomVJPPrimal(value=1)
    assert obj.value == 1


def test_class_SymbolicZero() -> None:
    """Test class SymbolicZero."""
    obj = mod.SymbolicZero(value=0)
    assert obj.value == 0


def test_closure_convert() -> None:
    """Test closure_convert."""
    with pytest.raises(AttributeError):
        mod.closure_convert()


def test_custom_gradient() -> None:
    """Test custom_gradient."""
    with pytest.raises(TypeError):
        mod.custom_gradient()


def test_class_custom_jvp() -> None:
    """Test class custom_jvp."""
    obj = mod.custom_jvp(name="cjvp")
    assert obj.name == "cjvp"


def test_custom_jvp_call_jaxpr_p() -> None:
    """Test custom_jvp_call_jaxpr_p."""
    assert mod.custom_jvp_call_jaxpr_p() is None


def test_custom_jvp_call_p() -> None:
    """Test custom_jvp_call_p."""
    assert mod.custom_jvp_call_p() is None


def test_class_custom_vjp() -> None:
    """Test class custom_vjp."""
    obj = mod.custom_vjp(name="cvjp")
    assert obj.name == "cvjp"


def test_custom_vjp_call_jaxpr_p() -> None:
    """Test custom_vjp_call_jaxpr_p."""
    assert mod.custom_vjp_call_jaxpr_p() is None


def test_custom_vjp_call_p() -> None:
    """Test custom_vjp_call_p."""
    assert mod.custom_vjp_call_p() is None


def test_custom_vjp_primal_tree_values() -> None:
    """Test custom_vjp_primal_tree_values."""
    with pytest.raises(AttributeError):
        mod.custom_vjp_primal_tree_values()


def test_linear_call() -> None:
    """Test linear_call."""
    with pytest.raises(AttributeError):
        mod.linear_call()
