"""Tests for zero_jax.interpreters.traceback_util.util.config.xla_client.hlo_sharding_util."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.util.config.xla_client.hlo_sharding_util as mod


def test_hlo_sharding_util() -> None:
    """Verify PartiallyReplicateTiledShardingOnDims."""
    obj = mod.PartiallyReplicateTiledShardingOnDims("tiled", (0, 1))
    assert obj.sharding == "tiled"
    assert obj.dims == (0, 1)
