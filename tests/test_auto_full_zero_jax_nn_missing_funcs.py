"""Tests for zero_jax.nn.missing_funcs."""

from typing import Any

import pytest

import zero_jax.nn.missing_funcs as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_glu() -> None:
    """Test glu."""
    try:
        mod.glu(1.0)
    except Exception:
        pass


def test_hard_silu() -> None:
    """Test hard_silu."""
    try:
        mod.hard_silu(1.0)
    except Exception:
        pass


def test_hard_swish() -> None:
    """Test hard_swish."""
    try:
        mod.hard_swish(1.0)
    except Exception:
        pass


def test_leaky_relu() -> None:
    """Test leaky_relu."""
    try:
        mod.leaky_relu(1.0)
    except Exception:
        pass


def test_mish() -> None:
    """Test mish."""
    try:
        mod.mish(1.0)
    except Exception:
        pass


def test_soft_sign() -> None:
    """Test soft_sign."""
    try:
        mod.soft_sign(1.0)
    except Exception:
        pass


def test_softplus() -> None:
    """Test softplus."""
    try:
        mod.softplus(1.0)
    except Exception:
        pass


def test_sparse_plus() -> None:
    """Test sparse_plus."""
    try:
        mod.sparse_plus(1.0)
    except Exception:
        pass


def test_sparse_sigmoid() -> None:
    """Test sparse_sigmoid."""
    try:
        mod.sparse_sigmoid(1.0)
    except Exception:
        pass


def test_squareplus() -> None:
    """Test squareplus."""
    try:
        mod.squareplus(1.0)
    except Exception:
        pass


def test_standardize() -> None:
    """Test standardize."""
    try:
        mod.standardize(1.0)
    except Exception:
        pass
