"""Frontend API routing for jax.scipy.cluster.vq."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def vq(*args: Any, **kwargs: Any) -> Any:
    """Assign codes from a code book to a set of observations."""
    return _ops.vq(*args, **kwargs)
