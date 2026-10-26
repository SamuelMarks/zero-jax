"""FFI submodule for zero_jax.extend."""

from __future__ import annotations


def stub(target_name: str = "ffi") -> str:
    """Return the registered FFI target name.

    Args:
        target_name: Foreign function interface target identifier.

    Returns:
        str: Target identifier.
    """
    return target_name


__all__ = ["stub"]
