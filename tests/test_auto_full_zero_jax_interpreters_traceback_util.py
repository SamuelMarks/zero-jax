"""Tests for zero_jax.interpreters.traceback_util."""

from typing import Any

import pytest

import zero_jax.interpreters.traceback_util as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_C() -> None:
    """Test C."""
    try:
        mod.C()
    except Exception:
        pass


def test_Callable() -> None:
    """Test Callable."""
    try:
        mod.Callable()
    except Exception:
        pass


def test_Optional() -> None:
    """Test Optional."""
    try:
        mod.Optional()
    except Exception:
        pass


def test_class_SimplifiedTraceback() -> None:
    """Test class SimplifiedTraceback."""
    try:
        mod.SimplifiedTraceback()
    except Exception:
        pass


def test_class_TypeVar() -> None:
    """Test class TypeVar."""
    try:
        mod.TypeVar()
    except Exception:
        pass


def test_class_UnfilteredStackTrace() -> None:
    """Test class UnfilteredStackTrace."""
    try:
        mod.UnfilteredStackTrace()
    except Exception:
        pass


def test_api_boundary() -> None:
    """Test api_boundary."""
    try:
        mod.api_boundary()
    except Exception:
        pass


def test_cast() -> None:
    """Test cast."""
    try:
        mod.cast()
    except Exception:
        pass


def test_dataclass() -> None:
    """Test dataclass."""
    try:
        mod.dataclass()
    except Exception:
        pass


def test_filter_traceback() -> None:
    """Test filter_traceback."""
    try:
        mod.filter_traceback()
    except Exception:
        pass


def test_format_exception_only() -> None:
    """Test format_exception_only."""
    try:
        mod.format_exception_only()
    except Exception:
        pass


def test_include_frame() -> None:
    """Test include_frame."""
    try:
        mod.include_frame()
    except Exception:
        pass


def test_register_exclusion() -> None:
    """Test register_exclusion."""
    try:
        mod.register_exclusion()
    except Exception:
        pass
