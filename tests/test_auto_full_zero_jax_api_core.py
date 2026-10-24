"""Tests for zero_jax.api.core."""

from __future__ import annotations

import zero_jax.api.core as mod


def test_ShapeDtypeStruct_Shard_NamedSharding() -> None:
    """Verify ShapeDtypeStruct, Shard, and NamedSharding."""
    sds = mod.ShapeDtypeStruct((2, 3), "float32")
    assert sds.shape == (2, 3)
    assert sds.dtype == "float32"

    shard = mod.Shard("cpu:0", (slice(None),))
    assert shard.device == "cpu:0"
    assert shard.index == (slice(None),)

    ns = mod.NamedSharding(None, None)
    assert ns.is_fully_replicated is True


def test_device_and_host_functions() -> None:
    """Verify device, backend, process, and host query functions."""
    assert mod.block_until_ready(42) == 42
    assert mod.default_backend() == "cpu"
    assert mod.device_count() == 1
    assert mod.device_put(7) == 7
    assert mod.device_put_replicated(3, ["cpu:0", "cpu:1"]) == [3, 3]
    assert mod.device_put_sharded([1, 2], ["cpu:0", "cpu:1"]).shape == (2,)
    assert mod.host_count() == 1
    assert mod.host_id() == 0
    assert mod.host_ids() == [0]
    assert mod.local_device_count() == 1
    assert mod.process_count() == 1
    assert mod.process_index() == 0
