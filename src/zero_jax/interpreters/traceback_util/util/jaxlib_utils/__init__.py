"""Frontend API routing for jax.interpreters.traceback_util.util.jaxlib_utils."""

from __future__ import annotations

from zero_jax.util import safe_map, safe_zip


class CallOp:
    """Descriptor for a low-level jaxlib call operation."""

    def __init__(self, name: str = "call") -> None:
        """Initialize ``CallOp``.

        Args:
            name: Operation name identifier.
        """
        self.name: str = str(name)


__all__ = [
    "CallOp",
    "safe_map",
    "safe_zip",
]
