"""Frontend API routing for jax.version."""

from __future__ import annotations

from . import datetime, os, pathlib, subprocess

__version__: str = "0.0.5"
__version_info__: tuple[int, int, int] = (0, 0, 5)


def annotations(name: str = "__future__.annotations") -> str:
    """Return the canonical annotation feature name for ``jax.version``.

    Args:
        name: Annotation feature identifier string.

    Returns:
        str: The canonical annotation feature identifier.
    """
    return name


__all__ = [
    "__version__",
    "__version_info__",
    "annotations",
    "datetime",
    "os",
    "pathlib",
    "subprocess",
]
