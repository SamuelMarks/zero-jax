"""Tests for zero_jax.errors."""

from __future__ import annotations

import pytest

import zero_jax.errors as mod


def test_class_ConcretizationTypeError() -> None:
    """Test class ConcretizationTypeError."""
    err = mod.ConcretizationTypeError("msg")
    assert isinstance(err, TypeError)
    with pytest.raises(mod.ConcretizationTypeError):
        raise err


def test_class_JAXIndexError() -> None:
    """Test class JAXIndexError."""
    err = mod.JAXIndexError("msg")
    assert isinstance(err, IndexError)
    with pytest.raises(mod.JAXIndexError):
        raise err


def test_class_JAXTypeError() -> None:
    """Test class JAXTypeError."""
    err = mod.JAXTypeError("msg")
    assert isinstance(err, TypeError)
    with pytest.raises(mod.JAXTypeError):
        raise err


def test_class_KeyReuseError() -> None:
    """Test class KeyReuseError."""
    err = mod.KeyReuseError("msg")
    assert isinstance(err, RuntimeError)
    with pytest.raises(mod.KeyReuseError):
        raise err


def test_class_NonConcreteBooleanIndexError() -> None:
    """Test class NonConcreteBooleanIndexError."""
    err = mod.NonConcreteBooleanIndexError("msg")
    assert isinstance(err, IndexError)
    with pytest.raises(mod.NonConcreteBooleanIndexError):
        raise err


def test_class_SimplifiedTraceback() -> None:
    """Test class SimplifiedTraceback."""
    tb = mod.SimplifiedTraceback(frames=["f1"])
    assert tb.frames == ["f1"]


def test_class_TracerArrayConversionError() -> None:
    """Test class TracerArrayConversionError."""
    err = mod.TracerArrayConversionError("msg")
    assert isinstance(err, TypeError)
    with pytest.raises(mod.TracerArrayConversionError):
        raise err


def test_class_TracerBoolConversionError() -> None:
    """Test class TracerBoolConversionError."""
    err = mod.TracerBoolConversionError("msg")
    assert isinstance(err, TypeError)
    with pytest.raises(mod.TracerBoolConversionError):
        raise err


def test_class_TracerIntegerConversionError() -> None:
    """Test class TracerIntegerConversionError."""
    err = mod.TracerIntegerConversionError("msg")
    assert isinstance(err, TypeError)
    with pytest.raises(mod.TracerIntegerConversionError):
        raise err


def test_class_UnexpectedTracerError() -> None:
    """Test class UnexpectedTracerError."""
    err = mod.UnexpectedTracerError("msg")
    assert isinstance(err, RuntimeError)
    with pytest.raises(mod.UnexpectedTracerError):
        raise err
