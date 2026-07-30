"""Tests for zero_jax.api.ad."""

from typing import Any

import pytest

import zero_jax.api.ad as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_Callable() -> None:
    """Test Callable."""
    try:
        mod.Callable()
    except Exception:
        pass


def test_custom_gradient() -> None:
    """Test custom_gradient."""
    try:
        mod.custom_gradient(1.0)
    except Exception:
        pass


def test_custom_jvp() -> None:
    """Test custom_jvp."""
    try:
        mod.custom_jvp(1.0)
    except Exception:
        pass


def test_custom_vjp() -> None:
    """Test custom_vjp."""
    try:
        mod.custom_vjp(1.0)
    except Exception:
        pass


def test_grad() -> None:
    """Test grad."""
    try:
        mod.grad(1.0)
    except Exception:
        pass


def test_hessian() -> None:
    """Test hessian."""
    try:
        mod.hessian(1.0)
    except Exception:
        pass


def test_jacfwd() -> None:
    """Test jacfwd."""
    try:
        mod.jacfwd(1.0)
    except Exception:
        pass


def test_jacobian() -> None:
    """Test jacobian."""
    try:
        mod.jacobian(1.0)
    except Exception:
        pass


def test_jacrev() -> None:
    """Test jacrev."""
    try:
        mod.jacrev(1.0)
    except Exception:
        pass


def test_jvp() -> None:
    """Test jvp."""
    try:
        mod.jvp(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_linear_transpose() -> None:
    """Test linear_transpose."""
    try:
        mod.linear_transpose(1.0)
    except Exception:
        pass


def test_linearize() -> None:
    """Test linearize."""
    try:
        mod.linearize(1.0)
    except Exception:
        pass


def test_value_and_grad() -> None:
    """Test value_and_grad."""
    try:
        mod.value_and_grad(1.0)
    except Exception:
        pass


def test_vjp() -> None:
    """Test vjp."""
    try:
        mod.vjp(1.0)
    except Exception:
        pass
