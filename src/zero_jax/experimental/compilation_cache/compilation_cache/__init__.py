"""Frontend API routing for jax.experimental.compilation_cache.compilation_cache."""

from __future__ import annotations

_CACHE_DIR: list[str | None] = [None]


def initialize_cache(path: str, max_cache_size_bytes: int = -1) -> None:
    """Initialize the persistent compilation cache directory (deprecated; prefer ``set_cache_dir``).

    Args:
        path: Filesystem path for cached compiled executables.
        max_cache_size_bytes: Optional maximum cache size in bytes.
    """
    _ = max_cache_size_bytes
    _CACHE_DIR[0] = str(path)


def is_initialized() -> bool:
    """Return whether the compilation cache directory has been initialized.

    Returns:
        bool: ``True`` if a cache directory is configured, else ``False``.
    """
    return _CACHE_DIR[0] is not None


def reset_cache() -> None:
    """Reset the compilation cache back to its uninitialized state."""
    _CACHE_DIR[0] = None


def set_cache_dir(path: str) -> None:
    """Set the persistent compilation cache directory.

    Args:
        path: Filesystem path for cached compiled executables.
    """
    _CACHE_DIR[0] = str(path)


__all__ = ["initialize_cache", "is_initialized", "reset_cache", "set_cache_dir"]
