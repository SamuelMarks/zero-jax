"""Backend submodule for zero_jax.extend."""

from __future__ import annotations

from zero_jax.api.core import default_backend


def stub(platform: str | None = None) -> str:
    """Return the active or requested backend platform identifier.

    Args:
        platform: Optional backend platform override.

    Returns:
        str: Backend platform name.
    """
    return platform if platform is not None else default_backend()


__all__ = ["stub"]
