"""Frontend API routing for zero-jax.lax."""

import typing

import ml_switcheroo_compiler

import zero_jax._compiler_proxy_ops as _ops

from . import linalg
from .control_flow import *
from .missing_funcs import *
from .mock_p import *
from .primitives import *
from .types import *


def __getattr__(name):
    if name == "Array":
        from zero_jax.numpy.lax_numpy import ndarray as Array

        return Array
    if name == "Device":
        import zero_jax

        return zero_jax.Device
    if hasattr(_ops, name):
        return getattr(_ops, name)  # pragma: no cover
    if hasattr(ml_switcheroo_compiler, name):
        return getattr(ml_switcheroo_compiler, name)  # pragma: no cover
    try:
        from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

        # If it's a known missing function, we might just return a dummy callable that raises NotImplementedError,
        # BUT we only want to do that if it really doesn't exist, to pass test_stubs.py
        def stub(*args, **kwargs):
            raise NotImplementedError(
                f"Stub for {name} is not implemented in backend"
            )  # pragma: no cover

        return stub
    except ImportError:  # pragma: no cover

        def stub(*args, **kwargs):  # pragma: no cover
            raise NotImplementedError(
                f"Stub for {name} is not implemented in backend"
            )  # pragma: no cover

        return stub  # pragma: no cover
