"""Print options configuration."""

from __future__ import annotations

import contextlib
from typing import Any

np = __import__("numpy")


def set_printoptions(*args: Any, **kwargs: Any) -> None:
    """Set print options."""
    np.set_printoptions(*args, **kwargs)


@contextlib.contextmanager
def printoptions(*args: Any, **kwargs: Any) -> Any:
    """Context manager for setting print options."""
    with np.printoptions(*args, **kwargs):
        yield
