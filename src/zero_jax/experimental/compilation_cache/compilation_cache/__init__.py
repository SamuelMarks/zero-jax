"""Frontend API routing for jax.experimental.compilation_cache.compilation_cache."""

from typing import Any


def initialize_cache(*args: Any, **kwargs: Any) -> Any:
    """This API is deprecated; use set_cache_dir instead."""
    raise NotImplementedError("initialize_cache not yet implemented in zero-jax")


def is_initialized(*args: Any, **kwargs: Any) -> Any:
    """Deprecated."""
    raise NotImplementedError("is_initialized not yet implemented in zero-jax")


def reset_cache(*args: Any, **kwargs: Any) -> Any:
    """Get back to pristine, uninitialized state."""
    raise NotImplementedError("reset_cache not yet implemented in zero-jax")


def set_cache_dir(*args: Any, **kwargs: Any) -> Any:
    """Sets the persistent compilation cache directory."""
    raise NotImplementedError("set_cache_dir not yet implemented in zero-jax")
