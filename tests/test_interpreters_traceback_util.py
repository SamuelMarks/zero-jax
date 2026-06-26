"""Tests for zero_jax module."""

import pytest
import zero_jax.interpreters.traceback_util as mod


def test_SimplifiedTraceback() -> None:
    """Test SimplifiedTraceback."""
    obj = mod.SimplifiedTraceback()
    assert obj is not None


def test_TypeVar() -> None:
    """Test TypeVar."""
    obj = mod.TypeVar()
    assert obj is not None


def test_UnfilteredStackTrace() -> None:
    """Test UnfilteredStackTrace."""
    obj = mod.UnfilteredStackTrace()
    assert obj is not None


def test_api_boundary() -> None:
    """Test api_boundary."""
    with pytest.raises(NotImplementedError):
        mod.api_boundary()


def test_cast() -> None:
    """Test cast."""
    with pytest.raises(NotImplementedError):
        mod.cast()


def test_filter_traceback() -> None:
    """Test filter_traceback."""
    with pytest.raises(NotImplementedError):
        mod.filter_traceback()


def test_format_exception_only() -> None:
    """Test format_exception_only."""
    with pytest.raises(NotImplementedError):
        mod.format_exception_only()


def test_include_frame() -> None:
    """Test include_frame."""
    with pytest.raises(NotImplementedError):
        mod.include_frame()


def test_register_exclusion() -> None:
    """Test register_exclusion."""
    with pytest.raises(NotImplementedError):
        mod.register_exclusion()
