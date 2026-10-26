"""Core submodule for zero_jax.extend."""

from __future__ import annotations


def stub(name: str = "core") -> str:
    """Return the extend.core primitive namespace identifier.

    Args:
        name: Optional primitive or namespace label.

    Returns:
        str: Namespace label.
    """
    return name


__all__ = ["stub"]
