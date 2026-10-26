"""Frontend API routing for hlo_sharding_util."""

from __future__ import annotations

from zero_jax.interpreters.traceback_util.config.xla_client.hlo_sharding_util import (
    PartiallyReplicateTiledShardingOnDims,
)

__all__ = [
    "PartiallyReplicateTiledShardingOnDims",
]
