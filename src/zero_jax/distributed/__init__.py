"""Frontend API routing for jax.distributed."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def initialize(*args: Any, **kwargs: Any) -> Any:
    """Initializes the JAX distributed system."""
    return _ops.initialize(*args, **kwargs)


def shutdown(*args: Any, **kwargs: Any) -> Any:
    """Shuts down the distributed system."""
    return _ops.shutdown(*args, **kwargs)
