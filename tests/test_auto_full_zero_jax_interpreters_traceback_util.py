"""Tests for zero_jax.interpreters.traceback_util."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util as mod


def test_C() -> None:
    """Test C."""
    assert mod.C() == "C"


def test_Callable() -> None:
    """Test Callable."""
    assert mod.Callable is not None


def test_Optional() -> None:
    """Test Optional."""
    assert mod.Optional is not None


def test_class_SimplifiedTraceback() -> None:
    """Test class SimplifiedTraceback."""
    obj = mod.SimplifiedTraceback()
    assert obj.name == "SimplifiedTraceback"


def test_class_TypeVar() -> None:
    """Test class TypeVar."""
    obj = mod.TypeVar("T")
    assert obj.__name__ == "T"


def test_class_UnfilteredStackTrace() -> None:
    """Test class UnfilteredStackTrace."""
    obj = mod.UnfilteredStackTrace()
    assert obj.name == "UnfilteredStackTrace"


def test_api_boundary() -> None:
    """Test api_boundary."""
    fn = lambda x: x + 1
    assert mod.api_boundary(fn)(2) == 3


def test_cast() -> None:
    """Test cast."""
    res = mod.cast("float32", 1.0)
    assert res == 1.0


def test_dataclass() -> None:
    """Test dataclass."""
    assert callable(mod.dataclass)


def test_filter_traceback() -> None:
    """Test filter_traceback."""
    assert mod.filter_traceback("trace") == "trace"


def test_format_exception_only() -> None:
    """Test format_exception_only."""
    assert mod.format_exception_only(RuntimeError, "boom") == ["RuntimeError: boom\n"]


def test_include_frame() -> None:
    """Test include_frame."""
    assert mod.include_frame("user_code.py") is True


def test_register_exclusion() -> None:
    """Test register_exclusion."""
    assert mod.register_exclusion("/some/internal/dir") == "/some/internal/dir"
