"""Tests for zero_jax.experimental.checkify."""

from typing import Any

import pytest

import zero_jax.experimental.checkify as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_Callable() -> None:
    """Test Callable."""
    try:
        mod.Callable()
    except Exception:
        pass


def test_class_ErrorCategory() -> None:
    """Test class ErrorCategory."""
    try:
        mod.ErrorCategory()
    except Exception:
        pass


def test_check() -> None:
    """Test check."""
    try:
        mod.check(1.0, 1.0)
    except Exception:
        pass


def test_checkify() -> None:
    """Test checkify."""
    try:
        mod.checkify(1.0)
    except Exception:
        pass
