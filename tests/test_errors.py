"""Tests for zero_jax.errors."""

from __future__ import annotations

import pytest

import zero_jax.errors as mod


def test_ConcretizationTypeError() -> None:
    """Verify ConcretizationTypeError is a TypeError and Exception."""
    err = mod.ConcretizationTypeError("concrete required")
    assert isinstance(err, TypeError)
    assert err.message == "concrete required"
    with pytest.raises(mod.ConcretizationTypeError, match="concrete required"):
        raise err


def test_JAXIndexError() -> None:
    """Verify JAXIndexError is an IndexError and Exception."""
    err = mod.JAXIndexError("bad index")
    assert isinstance(err, IndexError)
    with pytest.raises(mod.JAXIndexError, match="bad index"):
        raise err


def test_JAXTypeError() -> None:
    """Verify JAXTypeError is a TypeError and Exception."""
    err = mod.JAXTypeError("bad type")
    assert isinstance(err, TypeError)
    with pytest.raises(mod.JAXTypeError, match="bad type"):
        raise err


def test_KeyReuseError() -> None:
    """Verify KeyReuseError is a RuntimeError and Exception."""
    err = mod.KeyReuseError("key reused")
    assert isinstance(err, RuntimeError)
    with pytest.raises(mod.KeyReuseError, match="key reused"):
        raise err


def test_NonConcreteBooleanIndexError() -> None:
    """Verify NonConcreteBooleanIndexError is an IndexError and Exception."""
    err = mod.NonConcreteBooleanIndexError("non-concrete bool")
    assert isinstance(err, IndexError)
    with pytest.raises(mod.NonConcreteBooleanIndexError, match="non-concrete bool"):
        raise err


def test_SimplifiedTraceback() -> None:
    """Verify SimplifiedTraceback stores frames and works as a context manager."""
    with mod.SimplifiedTraceback(frames=["frame0"]) as obj:
        assert obj.frames == ["frame0"]


def test_TracerArrayConversionError() -> None:
    """Verify TracerArrayConversionError is a TypeError and Exception."""
    err = mod.TracerArrayConversionError("tracer array")
    assert isinstance(err, TypeError)
    with pytest.raises(mod.TracerArrayConversionError, match="tracer array"):
        raise err


def test_TracerBoolConversionError() -> None:
    """Verify TracerBoolConversionError is a TypeError and Exception."""
    err = mod.TracerBoolConversionError("tracer bool")
    assert isinstance(err, TypeError)
    with pytest.raises(mod.TracerBoolConversionError, match="tracer bool"):
        raise err


def test_TracerIntegerConversionError() -> None:
    """Verify TracerIntegerConversionError is a TypeError and Exception."""
    err = mod.TracerIntegerConversionError("tracer int")
    assert isinstance(err, TypeError)
    with pytest.raises(mod.TracerIntegerConversionError, match="tracer int"):
        raise err


def test_UnexpectedTracerError() -> None:
    """Verify UnexpectedTracerError is a RuntimeError and Exception."""
    err = mod.UnexpectedTracerError("leaked tracer")
    assert isinstance(err, RuntimeError)
    with pytest.raises(mod.UnexpectedTracerError, match="leaked tracer"):
        raise err
