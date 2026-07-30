"""Frontend API routing for jax.scipy.fft."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def dct(*args: Any, **kwargs: Any) -> Any:
    """Computes the discrete cosine transform of the input"""
    return _ops.dct(*args, **kwargs)


def dctn(*args: Any, **kwargs: Any) -> Any:
    """Computes the multidimensional discrete cosine transform of the input"""
    return _ops.dctn(*args, **kwargs)


def idct(*args: Any, **kwargs: Any) -> Any:
    """Computes the inverse discrete cosine transform of the input"""
    return _ops.idct(*args, **kwargs)


def idctn(*args: Any, **kwargs: Any) -> Any:
    """Computes the multidimensional inverse discrete cosine transform of the input"""
    return _ops.idctn(*args, **kwargs)
