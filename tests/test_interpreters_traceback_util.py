"""Tests for zero_jax.interpreters.traceback_util."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util as mod


def test_Callable() -> None:
    """Test Callable export."""
    assert callable(mod.C)
    assert mod.C() == "C"
    assert mod.C("fn") == "fn"


def test_SimplifiedTraceback() -> None:
    """Test SimplifiedTraceback."""
    obj = mod.SimplifiedTraceback(id=1, name="test_SimplifiedTraceback", value="test_value")
    assert obj.id == 1
    assert obj.name == "test_SimplifiedTraceback"
    assert obj.value == "test_value"


def test_TypeVar() -> None:
    """Test TypeVar."""
    t_var = mod.TypeVar("T")
    assert t_var.__name__ == "T"


def test_UnfilteredStackTrace() -> None:
    """Test UnfilteredStackTrace."""
    obj = mod.UnfilteredStackTrace(id=1, name="test_UnfilteredStackTrace", value="test_value")
    assert obj.id == 1
    assert obj.name == "test_UnfilteredStackTrace"
    assert obj.value == "test_value"


def test_api_boundary() -> None:
    """Test api_boundary."""
    fn = lambda x: x + 1
    assert mod.api_boundary(fn) is fn
    decorator = mod.api_boundary()
    assert decorator(fn) is fn


def test_cast() -> None:
    """Test cast."""
    assert mod.cast(int, 3) == 3
    assert mod.cast(int) is int


def test_filter_traceback() -> None:
    """Test filter_traceback."""
    assert mod.filter_traceback("tb") == "tb"


def test_format_exception_only() -> None:
    """Test format_exception_only."""
    assert mod.format_exception_only(ValueError, "bad") == ["ValueError: bad\n"]


def test_include_frame() -> None:
    """Test include_frame."""
    mod.register_exclusion("/excluded/path")
    assert mod.include_frame("/included/path/file.py") is True
    assert mod.include_frame("/excluded/path/file.py") is False


def test_register_exclusion() -> None:
    """Test register_exclusion."""
    assert mod.register_exclusion("/custom/exclude") == "/custom/exclude"
    assert mod.register_exclusion("/custom/exclude") == "/custom/exclude"
