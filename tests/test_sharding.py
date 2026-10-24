"""Tests for zero_jax.sharding."""

from __future__ import annotations

import zero_jax
import zero_jax.sharding as mod


def test_Sharding_properties_and_equivalence() -> None:
    """Verify Sharding properties and equivalence methods."""
    s1 = mod.Sharding(devices=("cpu:0",), id=1, name="s", value="v")
    s2 = mod.Sharding(devices=("cpu:0",))
    s3 = mod.Sharding(devices=("cpu:0", "cpu:1"))

    assert s1.device_set == {"cpu:0"}
    assert s1.is_fully_addressable is True
    assert s1.is_fully_replicated is True
    assert s3.is_fully_replicated is False
    assert s1.is_equivalent_to(s2) is True
    assert s1.is_equivalent_to(s3) is False


def test_PartitionSpec_indexing_and_equality() -> None:
    """Verify PartitionSpec indexing, length, and equality."""
    ps1 = mod.PartitionSpec("x", None, ("y", "z"))
    ps2 = mod.PartitionSpec("x", None, ("y", "z"))
    ps3 = mod.PartitionSpec("y")

    assert len(ps1) == 3
    assert ps1[0] == "x"
    assert ps1[1] is None
    assert ps1[2] == ("y", "z")
    assert ps1 == ps2
    assert ps1 != ps3
    assert (ps1 == "not_a_spec") is False


def test_Mesh_and_NamedSharding() -> None:
    """Verify Mesh and NamedSharding behavior and canonical identity."""
    assert zero_jax.NamedSharding is mod.NamedSharding
    mesh = mod.Mesh(devices=("cpu:0", "cpu:1"), axis_names=("x",))
    assert mesh.shape == {"x": 2}
    empty_mesh = mod.Mesh(devices=(), axis_names=())
    assert empty_mesh.shape == {}

    spec_rep = mod.PartitionSpec(None, None)
    spec_part = mod.PartitionSpec("x", None)
    ns_rep = mod.NamedSharding(mesh, spec_rep)
    ns_part = mod.NamedSharding(mesh, spec_part)

    assert ns_rep.mesh is mesh
    assert ns_rep.spec is spec_rep
    assert ns_rep.is_fully_replicated is True
    assert ns_part.is_fully_replicated is False


def test_SingleDevice_Positional_Pmap_GSPMD_Sharding() -> None:
    """Verify SingleDeviceSharding, PositionalSharding, PmapSharding, and GSPMDSharding."""
    single = mod.SingleDeviceSharding("gpu:0")
    assert single.device == "gpu:0"
    assert single.device_set == {"gpu:0"}

    pos = mod.PositionalSharding(("cpu:0", "cpu:1"))
    pos_rep = pos.replicate()
    assert pos_rep.device_set == {"cpu:0", "cpu:1"}

    pmap_s = mod.PmapSharding(("cpu:0",), sharding_spec="spec0")
    assert pmap_s.devices == ("cpu:0",)
    assert pmap_s.sharding_spec == "spec0"

    gspmd = mod.GSPMDSharding(("cpu:0",), op_sharding="replicated")
    assert gspmd.op_sharding == "replicated"
