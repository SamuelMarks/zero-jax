"""Frontend API routing for jax.scipy.fft."""

from typing import Any


def dct(*args: Any, **kwargs: Any) -> Any:
    """Computes the discrete cosine transform of the input"""
    raise NotImplementedError("dct not yet implemented in zero-jax")


def dctn(*args: Any, **kwargs: Any) -> Any:
    """Computes the multidimensional discrete cosine transform of the input"""
    raise NotImplementedError("dctn not yet implemented in zero-jax")


def idct(*args: Any, **kwargs: Any) -> Any:
    """Computes the inverse discrete cosine transform of the input"""
    raise NotImplementedError("idct not yet implemented in zero-jax")


def idctn(*args: Any, **kwargs: Any) -> Any:
    """Computes the multidimensional inverse discrete cosine transform of the input"""
    raise NotImplementedError("idctn not yet implemented in zero-jax")
