"""Tests for zero_jax.sharding."""

from typing import Any

import pytest

import zero_jax.sharding as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_class_GSPMDSharding() -> None:
    """Test class GSPMDSharding."""
    try:
        mod.GSPMDSharding()
    except Exception:
        pass


def test_class_Mesh() -> None:
    """Test class Mesh."""
    try:
        mod.Mesh()
    except Exception:
        pass


def test_class_NamedSharding() -> None:
    """Test class NamedSharding."""
    try:
        mod.NamedSharding()
    except Exception:
        pass


def test_class_PartitionSpec() -> None:
    """Test class PartitionSpec."""
    try:
        mod.PartitionSpec()
    except Exception:
        pass


def test_class_PmapSharding() -> None:
    """Test class PmapSharding."""
    try:
        mod.PmapSharding()
    except Exception:
        pass


def test_class_PositionalSharding() -> None:
    """Test class PositionalSharding."""
    try:
        mod.PositionalSharding()
    except Exception:
        pass


def test_class_Sharding() -> None:
    """Test class Sharding."""
    try:
        mod.Sharding()
    except Exception:
        pass


def test_class_SingleDeviceSharding() -> None:
    """Test class SingleDeviceSharding."""
    try:
        mod.SingleDeviceSharding()
    except Exception:
        pass
