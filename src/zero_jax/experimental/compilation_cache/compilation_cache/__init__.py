"""Frontend API routing for jax.experimental.compilation_cache.compilation_cache."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def initialize_cache(*args: Any, **kwargs: Any) -> Any:
    """This API is deprecated; use set_cache_dir instead."""
    return getattr(_ops, "initialize_cache")(*args, **kwargs)


def is_initialized(*args: Any, **kwargs: Any) -> Any:
    """Deprecated."""
    return getattr(_ops, "is_initialized")(*args, **kwargs)


def reset_cache(*args: Any, **kwargs: Any) -> Any:
    """Get back to pristine, uninitialized state."""
    return getattr(_ops, "reset_cache")(*args, **kwargs)


def set_cache_dir(*args: Any, **kwargs: Any) -> Any:
    """Sets the persistent compilation cache directory."""
    return getattr(_ops, "set_cache_dir")(*args, **kwargs)
