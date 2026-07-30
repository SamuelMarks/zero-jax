"""Frontend API routing for jax.experimental.compilation_cache.compilation_cache."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def initialize_cache(*args: Any, **kwargs: Any) -> Any:
    """This API is deprecated; use set_cache_dir instead."""
    return _ops.initialize_cache(*args, **kwargs)  # pragma: no cover


def is_initialized(*args: Any, **kwargs: Any) -> Any:
    """Deprecated."""
    return _ops.is_initialized(*args, **kwargs)  # pragma: no cover


def reset_cache(*args: Any, **kwargs: Any) -> Any:
    """Get back to pristine, uninitialized state."""
    return _ops.reset_cache(*args, **kwargs)  # pragma: no cover


def set_cache_dir(*args: Any, **kwargs: Any) -> Any:
    """Sets the persistent compilation cache directory."""
    return _ops.set_cache_dir(*args, **kwargs)  # pragma: no cover
