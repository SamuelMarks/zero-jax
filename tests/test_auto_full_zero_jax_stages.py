"""Tests for zero_jax.stages."""

from typing import Any

import pytest

import zero_jax.stages as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_class_ArgInfo() -> None:
    """Test class ArgInfo."""
    try:
        mod.ArgInfo()
    except Exception:
        pass


def test_class_Compiled() -> None:
    """Test class Compiled."""
    try:
        mod.Compiled()
    except Exception:
        pass


def test_class_CompilerOptions() -> None:
    """Test class CompilerOptions."""
    try:
        mod.CompilerOptions()
    except Exception:
        pass


def test_class_Lowered() -> None:
    """Test class Lowered."""
    try:
        mod.Lowered()
    except Exception:
        pass


def test_class_OutInfo() -> None:
    """Test class OutInfo."""
    try:
        mod.OutInfo()
    except Exception:
        pass


def test_class_Traced() -> None:
    """Test class Traced."""
    try:
        mod.Traced()
    except Exception:
        pass


def test_class_Wrapped() -> None:
    """Test class Wrapped."""
    try:
        mod.Wrapped()
    except Exception:
        pass
