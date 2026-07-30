"""Tests for zero_jax module."""

from unittest.mock import patch

import pytest

import zero_jax.interpreters.traceback_util as mod


def test_Callable() -> None:
    """Test Callable export."""
    assert mod.Callable is not None


def test_SimplifiedTraceback() -> None:
    """Test SimplifiedTraceback."""
    obj = mod.SimplifiedTraceback(
        id=1, name="test_SimplifiedTraceback", value="test_value"
    )
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_SimplifiedTraceback"
    assert obj.value == "test_value"


def test_TypeVar() -> None:
    """Test TypeVar."""
    obj = mod.TypeVar(id=1, name="test_TypeVar", value="test_value")
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_TypeVar"
    assert obj.value == "test_value"


def test_UnfilteredStackTrace() -> None:
    """Test UnfilteredStackTrace."""
    obj = mod.UnfilteredStackTrace(
        id=1, name="test_UnfilteredStackTrace", value="test_value"
    )
    assert obj is not None
    assert obj.id == 1
    assert obj.name == "test_UnfilteredStackTrace"
    assert obj.value == "test_value"


def test_api_boundary() -> None:
    """Test api_boundary."""
    with patch("zero_jax._compiler_proxy_ops.api_boundary", create=True) as mock_op:
        mod.api_boundary()
        mock_op.assert_called_once_with()


def test_cast() -> None:
    """Test cast."""
    with patch("zero_jax._compiler_proxy_ops.cast", create=True) as mock_op:
        mod.cast()
        mock_op.assert_called_once_with()


def test_filter_traceback() -> None:
    """Test filter_traceback."""
    with patch("zero_jax._compiler_proxy_ops.filter_traceback", create=True) as mock_op:
        mod.filter_traceback()
        mock_op.assert_called_once_with()


def test_format_exception_only() -> None:
    """Test format_exception_only."""
    with patch(
        "zero_jax._compiler_proxy_ops.format_exception_only", create=True
    ) as mock_op:
        mod.format_exception_only()
        mock_op.assert_called_once_with()


def test_include_frame() -> None:
    """Test include_frame."""
    with patch("zero_jax._compiler_proxy_ops.include_frame", create=True) as mock_op:
        mod.include_frame()
        mock_op.assert_called_once_with()


def test_register_exclusion() -> None:
    """Test register_exclusion."""
    with patch(
        "zero_jax._compiler_proxy_ops.register_exclusion", create=True
    ) as mock_op:
        mod.register_exclusion()
        mock_op.assert_called_once_with()
