"""Frontend API routing for jax.experimental."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import AbstractContextManager

from zero_jax._config_internal import enable_x64 as _enable_x64
from zero_jax._types import ArrayLike, PyTree, VariadicCallable
from zero_jax.api.misc import pure_callback
from zero_jax.numpy.lax_numpy import asarray, ndarray

from . import compilation_cache, pallas, x64_context


class EArray:
    """Extensible array wrapper holding an underlying ``ndarray``."""

    def __init__(self, data: ArrayLike = 0.0, *, aval: PyTree = None) -> None:
        """Initialize ``EArray`` with backing array data.

        Args:
            data: Backing array-like data.
            aval: Optional abstract value descriptor.
        """
        self.data: ndarray = asarray(data)
        self.aval: PyTree = aval
        self.shape: tuple[int, ...] = self.data.shape
        self.dtype = self.data.dtype


def disable_x64() -> AbstractContextManager[None]:
    """Experimental context manager to temporarily disable X64 mode.

    Returns:
        AbstractContextManager[None]: Context manager disabling 64-bit precision.
    """
    return _enable_x64(False)


def enable_x64(new_val: bool = True) -> Generator[None, None, None]:
    """Experimental context manager to temporarily enable X64 mode.

    Args:
        new_val: Whether 64-bit mode should be enabled inside the context.

    Returns:
        Generator[None, None, None]: Context manager enabling 64-bit precision.
    """
    return _enable_x64(bool(new_val))


def io_callback(
    callback: VariadicCallable[PyTree],
    result_shape_dtypes: PyTree,
    *args: PyTree,
    sharding: PyTree = None,
    ordered: bool = False,
    **kwargs: PyTree,
) -> PyTree:
    """Call an impure Python host callback with side-effect support.

    Args:
        callback: Host Python callable to invoke.
        result_shape_dtypes: PyTree of expected output ``ShapeDtypeStruct`` or example arrays.
        *args: Positional arguments forwarded to ``callback``.
        sharding: Optional sharding specification.
        ordered: Whether sequential ordering of IO callbacks is enforced.
        **kwargs: Keyword arguments forwarded to ``callback``.

    Returns:
        PyTree: Result of invoking ``callback``.
    """
    _ = (sharding, ordered)
    return pure_callback(callback, result_shape_dtypes, *args, **kwargs)


class BlurConfig:
    """Frontend class for ``BlurConfig``."""


class CTCLossOptions:
    """Frontend class for ``CTCLossOptions``."""


class ElasticConfig:
    """Frontend class for ``ElasticConfig``."""


class PerspectiveConfig:
    """Frontend class for ``PerspectiveConfig``."""


__all__ = [
    "BlurConfig",
    "CTCLossOptions",
    "EArray",
    "ElasticConfig",
    "PerspectiveConfig",
    "compilation_cache",
    "disable_x64",
    "enable_x64",
    "io_callback",
    "pallas",
    "x64_context",
]
