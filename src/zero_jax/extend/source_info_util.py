"""Source info util submodule for zero_jax.extend."""

from __future__ import annotations


def stub(source: str = "<unknown>") -> str:
    """Return a formatted source info frame label.

    Args:
        source: Source file or frame description.

    Returns:
        str: Source info label.
    """
    return source


__all__ = ["stub"]
