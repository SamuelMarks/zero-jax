"""Tests for zero_jax module."""

import pytest

import zero_jax.errors as mod


def test_ConcretizationTypeError() -> None:
    """Test ConcretizationTypeError."""
    obj = mod.ConcretizationTypeError()
    assert obj is not None


def test_JAXIndexError() -> None:
    """Test JAXIndexError."""
    obj = mod.JAXIndexError()
    assert obj is not None


def test_JAXTypeError() -> None:
    """Test JAXTypeError."""
    obj = mod.JAXTypeError()
    assert obj is not None


def test_KeyReuseError() -> None:
    """Test KeyReuseError."""
    obj = mod.KeyReuseError()
    assert obj is not None


def test_NonConcreteBooleanIndexError() -> None:
    """Test NonConcreteBooleanIndexError."""
    obj = mod.NonConcreteBooleanIndexError()
    assert obj is not None


def test_SimplifiedTraceback() -> None:
    """Test SimplifiedTraceback."""
    obj = mod.SimplifiedTraceback()
    assert obj is not None


def test_TracerArrayConversionError() -> None:
    """Test TracerArrayConversionError."""
    obj = mod.TracerArrayConversionError()
    assert obj is not None


def test_TracerBoolConversionError() -> None:
    """Test TracerBoolConversionError."""
    obj = mod.TracerBoolConversionError()
    assert obj is not None


def test_TracerIntegerConversionError() -> None:
    """Test TracerIntegerConversionError."""
    obj = mod.TracerIntegerConversionError()
    assert obj is not None


def test_UnexpectedTracerError() -> None:
    """Test UnexpectedTracerError."""
    obj = mod.UnexpectedTracerError()
    assert obj is not None
