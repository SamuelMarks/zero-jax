"""Tests for zero_jax.sharding."""

from __future__ import annotations

import zero_jax
import zero_jax.sharding as mod


def test_Sharding_properties_and_equivalence() -> None:
    """Verify Sharding properties and equivalence."""
    s1 = mod.Sharding(("cpu:0",))
    s2 = mod.Sharding(("cpu:0",))
    assert s1.device_set == {"cpu:0"}
    assert s1.is_fully_addressable is True
    assert s1.is_fully_replicated is True
    assert s1.is_equivalent_to(s2) is True


def test_PartitionSpec_and_Mesh() -> None:
    """Verify PartitionSpec indexing, equality, and Mesh shape."""
    ps1 = mod.PartitionSpec("x", "y")
    ps2 = mod.PartitionSpec("x", "y")
    assert len(ps1) == 2
    assert ps1[0] == "x"
    assert ps1 == ps2
    assert (ps1 == "invalid") is False

    mesh = mod.Mesh(devices=("cpu:0", "cpu:1"), axis_names=("x",))
    assert mesh.shape == {"x": 2}
    empty_mesh = mod.Mesh(devices=("cpu:0",), axis_names=())
    assert empty_mesh.shape == {}


def test_NamedSharding_SingleDevice_Positional_Pmap_GSPMD() -> None:
    """Verify NamedSharding, SingleDeviceSharding, PositionalSharding, PmapSharding, and GSPMDSharding."""
    mesh = mod.Mesh(("cpu:0", "cpu:1"), ("x",))
    ns_rep = mod.NamedSharding(mesh, mod.PartitionSpec(None, None))
    assert ns_rep.is_fully_replicated is True
    ns_part = mod.NamedSharding(mesh, mod.PartitionSpec("x", None))
    assert ns_part.is_fully_replicated is False

    sds = mod.make_single_device_sharding("cpu:0", memory_kind="device")
    assert sds.device == "cpu:0"
    assert sds.memory_kind == "device"
    assert isinstance(zero_jax.sharding.make_single_device_sharding("cpu:0"), mod.SingleDeviceSharding)

    pos = mod.PositionalSharding(("cpu:0",))
    assert isinstance(pos.replicate(), mod.PositionalSharding)

    pmap_s = mod.PmapSharding(("cpu:0",), sharding_spec="spec0")
    assert pmap_s.devices == ("cpu:0",)
    assert pmap_s.sharding_spec == "spec0"

    gspmd = mod.GSPMDSharding(("cpu:0",), op_sharding="replicated")
    assert gspmd.op_sharding == "replicated"


def test_mesh_contexts_and_axes_decorators() -> None:
    """Verify set_mesh, use_abstract_mesh, get_mesh, get_abstract_mesh, auto_axes, explicit_axes, no_collectives, and reshard."""
    m = mod.Mesh(("cpu:0", "cpu:1"), ("data",))
    with mod.set_mesh(m) as active:
        assert active is m
        assert mod.get_mesh() is m
    assert mod.get_mesh().size == 1

    am = mod.AbstractMesh(axis_sizes=(2, 4), axis_names=("x", "y"), abstract_device=mod.AbstractDevice("tpu", 2, "tpu"))
    assert am.size == 8
    with mod.use_abstract_mesh(am) as active_am:
        assert active_am is am
        assert mod.get_abstract_mesh() is am
    assert mod.get_abstract_mesh().size == 1

    assert mod.AxisType().Auto == "Auto"
    assert mod.ManualAxisType().args == ()
    assert mod.UnreducedKind().sum == "sum"

    assert mod.auto_axes(lambda x: x + 1)(2) == 3
    assert mod.auto_axes(axes="x")(lambda x: x + 1)(2) == 3
    assert mod.explicit_axes(lambda x: x + 1)(2) == 3
    assert mod.explicit_axes(axes="x")(lambda x: x + 1)(2) == 3

    with mod.no_collectives() as flag:
        assert flag is True

    assert mod.reshard(10, None) == 10
