"""Tests for zero_jax.interpreters.traceback_util.util.config.xla_client."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.util.config.xla_client as mod


def test_class_Client() -> None:
    """Test class Client."""
    obj = mod.Client()
    assert isinstance(obj, mod.Client)


def test_class_FftType() -> None:
    """Test class FftType."""
    obj = mod.FftType()
    assert isinstance(obj, mod.FftType)


def test_class_Frame() -> None:
    """Test class Frame."""
    obj = mod.Frame()
    assert isinstance(obj, mod.Frame)


def test_class_Layout() -> None:
    """Test class Layout."""
    obj = mod.Layout()
    assert isinstance(obj, mod.Layout)


def test_class_Mapping() -> None:
    """Test class Mapping."""
    obj = mod.Mapping()
    assert isinstance(obj, mod.Mapping)


def test_class_Memory() -> None:
    """Test class Memory."""
    obj = mod.Memory()
    assert isinstance(obj, mod.Memory)


def test_class_Shape() -> None:
    """Test class Shape."""
    obj = mod.Shape()
    assert isinstance(obj, mod.Shape)


def test_class_XlaOp() -> None:
    """Test class XlaOp."""
    obj = mod.XlaOp()
    assert isinstance(obj, mod.XlaOp)
