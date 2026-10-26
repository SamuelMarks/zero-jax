"""Random submodule for zero_jax.extend."""

from __future__ import annotations


def stub(impl: str = "threefry2x32") -> str:
    """Return the PRNG implementation identifier.

    Args:
        impl: PRNG implementation name.

    Returns:
        str: PRNG implementation name.
    """
    return impl


__all__ = ["stub"]
