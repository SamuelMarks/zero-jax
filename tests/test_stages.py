"""Tests for zero_jax.stages."""

from __future__ import annotations

import zero_jax.stages as mod


def test_ArgInfo() -> None:
    """Test ArgInfo."""
    obj = mod.ArgInfo(aval="float32[2]", donated=True, shape=(2,), dtype="float32")
    assert obj._aval == "float32[2]"
    assert obj.donated is True
    assert obj.shape == (2,)


def test_Compiled() -> None:
    """Test Compiled."""
    obj = mod.Compiled(fun=lambda x: x + 1)
    assert obj(4) == 5


def test_CompilerOptions() -> None:
    """Test CompilerOptions."""
    obj = mod.CompilerOptions(opt=1)
    assert obj["opt"] == 1


def test_Lowered() -> None:
    """Test Lowered."""
    obj = mod.Lowered(fun=lambda x: x * 2)
    assert obj.compile()(3) == 6


def test_OutInfo() -> None:
    """Test OutInfo."""
    obj = mod.OutInfo(shape=(2, 3), dtype="float32")
    assert obj.shape == (2, 3)


def test_Traced() -> None:
    """Test Traced."""
    obj = mod.Traced(fun=lambda x: x - 1)
    assert obj.lower().compile()(5) == 4


def test_Wrapped() -> None:
    """Test Wrapped."""
    obj = mod.Wrapped(fun=lambda x: x + 10)
    assert obj(2) == 12
