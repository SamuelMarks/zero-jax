"""Frontend API routing for jax.interpreters.traceback_util."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue


def C(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``C``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


@dataclass
class SimplifiedTraceback:
    """Data structure for SimplifiedTraceback.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "SimplifiedTraceback"
    value: ArrayLikeSpec | None = None


class TypeVar:
    """Type variable."""

    def __init__(self, *args: ConfigValue, id: ConfigValue = None, name: ConfigValue = None, value: ConfigValue = None, dummy: ConfigValue = None, **kwargs: ConfigValue) -> None:
        """Initialize instance with optional configuration attributes.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
            dummy: Optional test placeholder attribute.
            **kwargs: Additional keyword attributes.
        """
        self.args = args
        self.id = id
        self.name = name
        self.value = value
        self.dummy = dummy
        self.kwargs = kwargs


@dataclass
class UnfilteredStackTrace:
    """Data structure for UnfilteredStackTrace.

    Attributes:
        id (int): Identifier.
        name (str): Name of the object.
        value (Optional[ArrayLikeSpec]): Optional value.
    """

    id: int = 0
    name: str = "UnfilteredStackTrace"
    value: ArrayLikeSpec | None = None


annotations: ArrayLikeSpec = None


def api_boundary(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Wraps ``fun`` to form a boundary for filtering exception tracebacks.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.api_boundary(*args, **kwargs)


def cast(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Cast a value to a type.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.cast(*args, **kwargs)


from . import config


def filter_traceback(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for filter_traceback.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.filter_traceback(*args, **kwargs)


def format_exception_only(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for format_exception_only.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.format_exception_only(*args, **kwargs)


from . import functools


def include_frame(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for include_frame.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.include_frame(*args, **kwargs)


from . import os


def register_exclusion(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Frontend wrapper for register_exclusion.

    Args:
        *args (ArrayLikeSpec): Positional arguments.
        **kwargs (ArrayLikeSpec): Keyword arguments.

    Returns:
        ArrayLikeSpec: The result of the backend operation.
    """
    return _ops.register_exclusion(*args, **kwargs)


import typing
from collections.abc import Callable

import ml_switcheroo_compiler

from . import sys, traceback, types, util
