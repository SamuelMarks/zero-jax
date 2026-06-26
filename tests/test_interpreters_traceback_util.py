"""Tests for zero_jax module."""

import pytest
from unittest.mock import patch
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
    with patch("ml_switcheroo_compiler.ops.api_boundary") as mock_op:
        mod.api_boundary()
        mock_op.assert_called_once_with()


def test_cast() -> None:
    """Test cast."""
    with patch("ml_switcheroo_compiler.ops.cast") as mock_op:
        mod.cast()
        mock_op.assert_called_once_with()


def test_filter_traceback() -> None:
    """Test filter_traceback."""
    with patch("ml_switcheroo_compiler.ops.filter_traceback") as mock_op:
        mod.filter_traceback()
        mock_op.assert_called_once_with()


def test_format_exception_only() -> None:
    """Test format_exception_only."""
    with patch("ml_switcheroo_compiler.ops.format_exception_only") as mock_op:
        mod.format_exception_only()
        mock_op.assert_called_once_with()


def test_include_frame() -> None:
    """Test include_frame."""
    with patch("ml_switcheroo_compiler.ops.include_frame") as mock_op:
        mod.include_frame()
        mock_op.assert_called_once_with()


def test_register_exclusion() -> None:
    """Test register_exclusion."""
    with patch("ml_switcheroo_compiler.ops.register_exclusion") as mock_op:
        mod.register_exclusion()
        mock_op.assert_called_once_with()
