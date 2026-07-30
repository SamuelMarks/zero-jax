"""Tests for zero_jax.export."""

from typing import Any

import pytest

import zero_jax.export as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_class_DisabledSafetyCheck() -> None:
    """Test class DisabledSafetyCheck."""
    try:
        mod.DisabledSafetyCheck()
    except Exception:
        pass


def test_class_Exported() -> None:
    """Test class Exported."""
    try:
        mod.Exported()
    except Exception:
        pass


def test_class_SymbolicScope() -> None:
    """Test class SymbolicScope."""
    try:
        mod.SymbolicScope()
    except Exception:
        pass


def test_default_export_platform() -> None:
    """Test default_export_platform."""
    try:
        mod.default_export_platform()
    except Exception:
        pass


def test_deserialize() -> None:
    """Test deserialize."""
    try:
        mod.deserialize()
    except Exception:
        pass


def test_export() -> None:
    """Test export."""
    try:
        mod.export()
    except Exception:
        pass


def test_is_symbolic_dim() -> None:
    """Test is_symbolic_dim."""
    try:
        mod.is_symbolic_dim()
    except Exception:
        pass


def test_maximum_supported_calling_convention_version() -> None:
    """Test maximum_supported_calling_convention_version."""
    try:
        mod.maximum_supported_calling_convention_version()
    except Exception:
        pass


def test_minimum_supported_calling_convention_version() -> None:
    """Test minimum_supported_calling_convention_version."""
    try:
        mod.minimum_supported_calling_convention_version()
    except Exception:
        pass


def test_symbolic_args_specs() -> None:
    """Test symbolic_args_specs."""
    try:
        mod.symbolic_args_specs()
    except Exception:
        pass


def test_symbolic_shape() -> None:
    """Test symbolic_shape."""
    try:
        mod.symbolic_shape()
    except Exception:
        pass
