"""Tests for zero_jax.sharding."""

from __future__ import annotations

import zero_jax.sharding as mod


def test_class_GSPMDSharding() -> None:
    """Test class GSPMDSharding."""
    obj = mod.GSPMDSharding(("cpu:0",), op_sharding="op")
    assert obj.op_sharding == "op"


def test_class_Mesh() -> None:
    """Test class Mesh."""
    obj = mod.Mesh(("cpu:0", "cpu:1"), ("data",))
    assert obj.shape == {"data": 2}


def test_class_NamedSharding() -> None:
    """Test class NamedSharding."""
    mesh = mod.Mesh(("cpu:0",), ("x",))
    spec = mod.PartitionSpec("x")
    obj = mod.NamedSharding(mesh, spec)
    assert obj.mesh is mesh
    assert obj.spec == spec


def test_class_PartitionSpec() -> None:
    """Test class PartitionSpec."""
    obj = mod.PartitionSpec("x", None)
    assert len(obj) == 2
    assert obj[0] == "x"


def test_class_PmapSharding() -> None:
    """Test class PmapSharding."""
    obj = mod.PmapSharding(("cpu:0",), sharding_spec="pmap_spec")
    assert obj.sharding_spec == "pmap_spec"


def test_class_PositionalSharding() -> None:
    """Test class PositionalSharding."""
    obj = mod.PositionalSharding(("cpu:0",))
    assert obj.replicate().device_set == {"cpu:0"}


def test_class_Sharding() -> None:
    """Test class Sharding."""
    obj = mod.Sharding(("cpu:0",))
    assert obj.is_fully_addressable is True


def test_class_SingleDeviceSharding() -> None:
    """Test class SingleDeviceSharding."""
    obj = mod.SingleDeviceSharding("cpu:0")
    assert obj.device == "cpu:0"
