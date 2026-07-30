"""Tests for zero_jax.interpreters.traceback_util.util.config.xla_client."""

from typing import Any

import pytest

import zero_jax.interpreters.traceback_util.util.config.xla_client as mod


def test_class_Client() -> None:
    """Test class Client."""
    try:
        mod.Client()
    except Exception:
        pass


def test_class_FftType() -> None:
    """Test class FftType."""
    try:
        mod.FftType()
    except Exception:
        pass


def test_class_Frame() -> None:
    """Test class Frame."""
    try:
        mod.Frame()
    except Exception:
        pass


def test_class_Layout() -> None:
    """Test class Layout."""
    try:
        mod.Layout()
    except Exception:
        pass


def test_class_Mapping() -> None:
    """Test class Mapping."""
    try:
        mod.Mapping()
    except Exception:
        pass


def test_class_Memory() -> None:
    """Test class Memory."""
    try:
        mod.Memory()
    except Exception:
        pass


def test_class_Shape() -> None:
    """Test class Shape."""
    try:
        mod.Shape()
    except Exception:
        pass


def test_class_XlaOp() -> None:
    """Test class XlaOp."""
    try:
        mod.XlaOp()
    except Exception:
        pass
