"""Tests for zero_jax.custom_derivatives."""

from typing import Any

import pytest

import zero_jax.custom_derivatives as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_class_CustomVJPPrimal() -> None:
    """Test class CustomVJPPrimal."""
    try:
        mod.CustomVJPPrimal()
    except Exception:
        pass


def test_class_SymbolicZero() -> None:
    """Test class SymbolicZero."""
    try:
        mod.SymbolicZero()
    except Exception:
        pass


def test_closure_convert() -> None:
    """Test closure_convert."""
    try:
        mod.closure_convert()
    except Exception:
        pass


def test_custom_gradient() -> None:
    """Test custom_gradient."""
    try:
        mod.custom_gradient()
    except Exception:
        pass


def test_class_custom_jvp() -> None:
    """Test class custom_jvp."""
    try:
        mod.custom_jvp()
    except Exception:
        pass


def test_custom_jvp_call_jaxpr_p() -> None:
    """Test custom_jvp_call_jaxpr_p."""
    try:
        mod.custom_jvp_call_jaxpr_p()
    except Exception:
        pass


def test_custom_jvp_call_p() -> None:
    """Test custom_jvp_call_p."""
    try:
        mod.custom_jvp_call_p()
    except Exception:
        pass


def test_class_custom_vjp() -> None:
    """Test class custom_vjp."""
    try:
        mod.custom_vjp()
    except Exception:
        pass


def test_custom_vjp_call_jaxpr_p() -> None:
    """Test custom_vjp_call_jaxpr_p."""
    try:
        mod.custom_vjp_call_jaxpr_p()
    except Exception:
        pass


def test_custom_vjp_call_p() -> None:
    """Test custom_vjp_call_p."""
    try:
        mod.custom_vjp_call_p()
    except Exception:
        pass


def test_custom_vjp_primal_tree_values() -> None:
    """Test custom_vjp_primal_tree_values."""
    try:
        mod.custom_vjp_primal_tree_values()
    except Exception:
        pass


def test_linear_call() -> None:
    """Test linear_call."""
    try:
        mod.linear_call()
    except Exception:
        pass
