"""Tests for zero_jax module."""

import pytest
import zero_jax.stages as mod


def test_ArgInfo() -> None:
    """Test ArgInfo."""
    obj = mod.ArgInfo()
    assert obj is not None


def test_Compiled() -> None:
    """Test Compiled."""
    obj = mod.Compiled()
    assert obj is not None


def test_CompilerOptions() -> None:
    """Test CompilerOptions."""
    obj = mod.CompilerOptions()
    assert obj is not None


def test_Lowered() -> None:
    """Test Lowered."""
    obj = mod.Lowered()
    assert obj is not None


def test_OutInfo() -> None:
    """Test OutInfo."""
    obj = mod.OutInfo()
    assert obj is not None


def test_Traced() -> None:
    """Test Traced."""
    obj = mod.Traced()
    assert obj is not None


def test_Wrapped() -> None:
    """Test Wrapped."""
    obj = mod.Wrapped()
    assert obj is not None
