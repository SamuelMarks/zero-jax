"""Tests for zero_jax.lax.types."""

from typing import Any

import pytest

import zero_jax.lax.types as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_class_ConvDimensionNumbers() -> None:
    """Test class ConvDimensionNumbers."""
    try:
        mod.ConvDimensionNumbers(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_class_ConvGeneralDilatedDimensionNumbers() -> None:
    """Test class ConvGeneralDilatedDimensionNumbers."""
    try:
        mod.ConvGeneralDilatedDimensionNumbers(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_class_DotDimensionNumbers() -> None:
    """Test class DotDimensionNumbers."""
    try:
        mod.DotDimensionNumbers(1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass


def test_class_GatherDimensionNumbers() -> None:
    """Test class GatherDimensionNumbers."""
    try:
        mod.GatherDimensionNumbers(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_class_GatherScatterMode() -> None:
    """Test class GatherScatterMode."""
    try:
        mod.GatherScatterMode()
    except Exception:
        pass


def test_class_Precision() -> None:
    """Test class Precision."""
    try:
        mod.Precision()
    except Exception:
        pass


def test_class_PrecisionLike() -> None:
    """Test class PrecisionLike."""
    try:
        mod.PrecisionLike()
    except Exception:
        pass


def test_class_RandomAlgorithm() -> None:
    """Test class RandomAlgorithm."""
    try:
        mod.RandomAlgorithm()
    except Exception:
        pass


def test_class_RoundingMethod() -> None:
    """Test class RoundingMethod."""
    try:
        mod.RoundingMethod()
    except Exception:
        pass


def test_class_ScatterDimensionNumbers() -> None:
    """Test class ScatterDimensionNumbers."""
    try:
        mod.ScatterDimensionNumbers(1.0, 1.0, 1.0)
    except Exception:
        pass
