"""Tests for zero_jax.errors."""

from typing import Any

import pytest

import zero_jax.errors as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_class_ConcretizationTypeError() -> None:
    """Test class ConcretizationTypeError."""
    try:
        mod.ConcretizationTypeError()
    except Exception:
        pass


def test_class_JAXIndexError() -> None:
    """Test class JAXIndexError."""
    try:
        mod.JAXIndexError()
    except Exception:
        pass


def test_class_JAXTypeError() -> None:
    """Test class JAXTypeError."""
    try:
        mod.JAXTypeError()
    except Exception:
        pass


def test_class_KeyReuseError() -> None:
    """Test class KeyReuseError."""
    try:
        mod.KeyReuseError()
    except Exception:
        pass


def test_class_NonConcreteBooleanIndexError() -> None:
    """Test class NonConcreteBooleanIndexError."""
    try:
        mod.NonConcreteBooleanIndexError()
    except Exception:
        pass


def test_class_SimplifiedTraceback() -> None:
    """Test class SimplifiedTraceback."""
    try:
        mod.SimplifiedTraceback()
    except Exception:
        pass


def test_class_TracerArrayConversionError() -> None:
    """Test class TracerArrayConversionError."""
    try:
        mod.TracerArrayConversionError()
    except Exception:
        pass


def test_class_TracerBoolConversionError() -> None:
    """Test class TracerBoolConversionError."""
    try:
        mod.TracerBoolConversionError()
    except Exception:
        pass


def test_class_TracerIntegerConversionError() -> None:
    """Test class TracerIntegerConversionError."""
    try:
        mod.TracerIntegerConversionError()
    except Exception:
        pass


def test_class_UnexpectedTracerError() -> None:
    """Test class UnexpectedTracerError."""
    try:
        mod.UnexpectedTracerError()
    except Exception:
        pass
