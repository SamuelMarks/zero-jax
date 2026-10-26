"""Frontend API routing for hlo_sharding_util."""

from __future__ import annotations

from collections.abc import Sequence


class PartiallyReplicateTiledShardingOnDims:
    """Descriptor for partially replicating a tiled HLO sharding along specified dimensions."""

    def __init__(
        self,
        sharding: str = "replicated",
        dims: Sequence[int] = (),
    ) -> None:
        """Initialize ``PartiallyReplicateTiledShardingOnDims``.

        Args:
            sharding: Input HLO sharding representation.
            dims: Sequence of tile dimensions to replicate.
        """
        self.sharding: str = str(sharding)
        self.dims: tuple[int, ...] = tuple(int(d) for d in dims)


__all__ = [
    "PartiallyReplicateTiledShardingOnDims",
]
