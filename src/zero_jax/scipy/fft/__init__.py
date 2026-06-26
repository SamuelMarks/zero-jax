"""Frontend API routing for jax.scipy.fft."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def dct(*args: Any, **kwargs: Any) -> Any:
    """Computes the discrete cosine transform of the input"""
    return getattr(_ops, "dct")(*args, **kwargs)


def dctn(*args: Any, **kwargs: Any) -> Any:
    """Computes the multidimensional discrete cosine transform of the input"""
    return getattr(_ops, "dctn")(*args, **kwargs)


def idct(*args: Any, **kwargs: Any) -> Any:
    """Computes the inverse discrete cosine transform of the input"""
    return getattr(_ops, "idct")(*args, **kwargs)


def idctn(*args: Any, **kwargs: Any) -> Any:
    """Computes the multidimensional inverse discrete cosine transform of the input"""
    return getattr(_ops, "idctn")(*args, **kwargs)
