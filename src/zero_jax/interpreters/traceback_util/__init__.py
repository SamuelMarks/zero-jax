"""Frontend API routing for jax.interpreters.traceback_util."""

from dataclasses import dataclass
from typing import Any, Optional

import zero_jax._compiler_proxy_ops as _ops


def C(*args: Any, **kwargs: Any) -> Any:
    return None


@dataclass
class SimplifiedTraceback:
    """Data structure for SimplifiedTraceback.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "SimplifiedTraceback"
    value: Optional[Any] = None


class TypeVar:
    """Type variable."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass
class UnfilteredStackTrace:
    """Data structure for UnfilteredStackTrace.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[Any]): Optional value.
    """

    id: int = 0
    name: str = "UnfilteredStackTrace"
    value: Optional[Any] = None


annotations: Any = None


def api_boundary(*args: Any, **kwargs: Any) -> Any:
    """Wraps ``fun`` to form a boundary for filtering exception tracebacks."""
    return _ops.api_boundary(*args, **kwargs)


def cast(*args: Any, **kwargs: Any) -> Any:
    """Cast a value to a type."""
    return _ops.cast(*args, **kwargs)


from . import config


def filter_traceback(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for filter_traceback.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.filter_traceback(*args, **kwargs)


def format_exception_only(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for format_exception_only.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.format_exception_only(*args, **kwargs)


from . import functools


def include_frame(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for include_frame.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.include_frame(*args, **kwargs)


from . import os


def register_exclusion(*args: Any, **kwargs: Any) -> Any:
    """Frontend wrapper for register_exclusion.

    Args:
        *args (Any): Positional arguments.
        **kwargs (Any): Keyword arguments.

    Returns:
        Any: The result of the backend operation.
    """
    return _ops.register_exclusion(*args, **kwargs)


import typing
from typing import Callable

import ml_switcheroo_compiler

from . import sys, traceback, types, util


def __getattr__(name):
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
