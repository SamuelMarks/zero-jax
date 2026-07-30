"""Print options configuration."""

from __future__ import annotations

import contextlib
from typing import Any


def set_printoptions(*args: Any, **kwargs: Any) -> None:
    """Set print options."""


@contextlib.contextmanager
def printoptions(*args: Any, **kwargs: Any) -> Any:
    """Context manager for setting print options."""
    yield  # pragma: no cover
