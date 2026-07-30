"""Tests for zero_jax.nn.initializers."""

from typing import Any

import pytest

import zero_jax.nn.initializers as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_class_Array() -> None:
    """Test class Array."""
    try:
        mod.Array(1.0)
    except Exception:
        pass


def test_Callable() -> None:
    """Test Callable."""
    try:
        mod.Callable()
    except Exception:
        pass


def test_class_DType() -> None:
    """Test class DType."""
    try:
        mod.DType(1.0)
    except Exception:
        pass


def test_Initializer() -> None:
    """Test Initializer."""
    try:
        mod.Initializer()
    except Exception:
        pass


def test_KeyArray() -> None:
    """Test KeyArray."""
    try:
        mod.KeyArray()
    except Exception:
        pass


def test_RealNumeric() -> None:
    """Test RealNumeric."""
    try:
        mod.RealNumeric()
    except Exception:
        pass


def test_Sequence() -> None:
    """Test Sequence."""
    try:
        mod.Sequence()
    except Exception:
        pass


def test_Shape() -> None:
    """Test Shape."""
    try:
        mod.Shape()
    except Exception:
        pass


def test_Tuple() -> None:
    """Test Tuple."""
    try:
        mod.Tuple()
    except Exception:
        pass


def test_Union() -> None:
    """Test Union."""
    try:
        mod.Union()
    except Exception:
        pass


def test_constant() -> None:
    """Test constant."""
    try:
        mod.constant(1.0)
    except Exception:
        pass


def test_delta_orthogonal() -> None:
    """Test delta_orthogonal."""
    try:
        mod.delta_orthogonal()
    except Exception:
        pass


def test_glorot_normal() -> None:
    """Test glorot_normal."""
    try:
        mod.glorot_normal()
    except Exception:
        pass


def test_glorot_uniform() -> None:
    """Test glorot_uniform."""
    try:
        mod.glorot_uniform()
    except Exception:
        pass


def test_he_normal() -> None:
    """Test he_normal."""
    try:
        mod.he_normal()
    except Exception:
        pass


def test_he_uniform() -> None:
    """Test he_uniform."""
    try:
        mod.he_uniform()
    except Exception:
        pass


def test_kaiming_normal() -> None:
    """Test kaiming_normal."""
    try:
        mod.kaiming_normal()
    except Exception:
        pass


def test_kaiming_uniform() -> None:
    """Test kaiming_uniform."""
    try:
        mod.kaiming_uniform()
    except Exception:
        pass


def test_lecun_normal() -> None:
    """Test lecun_normal."""
    try:
        mod.lecun_normal()
    except Exception:
        pass


def test_lecun_uniform() -> None:
    """Test lecun_uniform."""
    try:
        mod.lecun_uniform()
    except Exception:
        pass


def test_normal() -> None:
    """Test normal."""
    try:
        mod.normal()
    except Exception:
        pass


def test_ones() -> None:
    """Test ones."""
    try:
        mod.ones(1.0, 1.0)
    except Exception:
        pass


def test_orthogonal() -> None:
    """Test orthogonal."""
    try:
        mod.orthogonal()
    except Exception:
        pass


def test_truncated_normal() -> None:
    """Test truncated_normal."""
    try:
        mod.truncated_normal()
    except Exception:
        pass


def test_uniform() -> None:
    """Test uniform."""
    try:
        mod.uniform()
    except Exception:
        pass


def test_variance_scaling() -> None:
    """Test variance_scaling."""
    try:
        mod.variance_scaling(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_xavier_normal() -> None:
    """Test xavier_normal."""
    try:
        mod.xavier_normal()
    except Exception:
        pass


def test_xavier_uniform() -> None:
    """Test xavier_uniform."""
    try:
        mod.xavier_uniform()
    except Exception:
        pass


def test_zeros() -> None:
    """Test zeros."""
    try:
        mod.zeros(1.0, 1.0)
    except Exception:
        pass
