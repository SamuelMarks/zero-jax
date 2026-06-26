"""Frontend API routing for jax.scipy.cluster.vq."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def vq(*args: Any, **kwargs: Any) -> Any:
    """Assign codes from a code book to a set of observations."""
    return getattr(_ops, "vq")(*args, **kwargs)
