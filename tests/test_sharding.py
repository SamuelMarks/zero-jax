"""Tests for zero_jax module."""

import pytest
import zero_jax.sharding as mod


def test_GSPMDSharding() -> None:
    """Test GSPMDSharding."""
    obj = mod.GSPMDSharding()
    assert obj is not None


def test_Mesh() -> None:
    """Test Mesh."""
    obj = mod.Mesh()
    assert obj is not None


def test_NamedSharding() -> None:
    """Test NamedSharding."""
    obj = mod.NamedSharding()
    assert obj is not None


def test_PartitionSpec() -> None:
    """Test PartitionSpec."""
    obj = mod.PartitionSpec()
    assert obj is not None


def test_PmapSharding() -> None:
    """Test PmapSharding."""
    obj = mod.PmapSharding()
    assert obj is not None


def test_PositionalSharding() -> None:
    """Test PositionalSharding."""
    obj = mod.PositionalSharding()
    assert obj is not None


def test_SingleDeviceSharding() -> None:
    """Test SingleDeviceSharding."""
    obj = mod.SingleDeviceSharding()
    assert obj is not None
