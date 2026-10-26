"""Frontend API routing for jax.interpreters.traceback_util."""

from __future__ import annotations

import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, TypeVar

import ml_switcheroo_compiler

from zero_jax._types import ArrayLikeSpec

from . import config, functools, os, sys, traceback, types, util

_EXCLUDED_PATHS: list[str] = []


def C(val: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Identity callable type variable placeholder ``C``.

    Args:
        val: Optional callable or value.

    Returns:
        ArrayLikeSpec: Unmodified ``val`` or ``"C"`` when ``None``.
    """
    return "C" if val is None else val


@dataclass
class SimplifiedTraceback:
    """Data structure for SimplifiedTraceback."""

    id: int = 0
    name: str = "SimplifiedTraceback"
    value: ArrayLikeSpec | None = None


@dataclass
class UnfilteredStackTrace:
    """Data structure for UnfilteredStackTrace."""

    id: int = 0
    name: str = "UnfilteredStackTrace"
    value: ArrayLikeSpec | None = None


annotations: str = "__future__.annotations"


def api_boundary(fun: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Wrap ``fun`` to form a boundary for filtering exception tracebacks.

    Args:
        fun: Function to wrap at the API boundary.

    Returns:
        ArrayLikeSpec: Wrapped function (or identity decorator when ``fun`` is ``None``).
    """
    if fun is None:
        return lambda f: f
    return fun


def cast(typ: ArrayLikeSpec = None, val: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return ``val`` unchanged as a runtime type cast helper.

    Args:
        typ: Target type descriptor.
        val: Value to cast.

    Returns:
        ArrayLikeSpec: ``val`` when provided, otherwise ``typ``.
    """
    return val if val is not None else typ


def filter_traceback(tb: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Filter internal JAX frames from traceback ``tb``.

    Args:
        tb: Input traceback object.

    Returns:
        ArrayLikeSpec: Filtered traceback object.
    """
    return tb


def format_exception_only(etype: type[BaseException] = Exception, value: BaseException | str = "") -> list[str]:
    """Format an exception type and value without a traceback.

    Args:
        etype: Exception class.
        value: Exception instance or message string.

    Returns:
        list[str]: Formatted exception lines.
    """
    return [f"{etype.__name__}: {value}\n"]


def include_frame(frame: ArrayLikeSpec = None) -> bool:
    """Return ``True`` if ``frame`` should be included in filtered tracebacks.

    Args:
        frame: Frame object or filename string.

    Returns:
        bool: ``True`` if the frame is not in ``_EXCLUDED_PATHS``.
    """
    frame_str = str(frame or "")
    return not any(excl and excl in frame_str for excl in _EXCLUDED_PATHS)


def register_exclusion(path: str = "") -> str:
    """Register a file or module ``path`` to exclude from filtered tracebacks.

    Args:
        path: File path prefix to exclude.

    Returns:
        str: Registered exclusion path.
    """
    if path and path not in _EXCLUDED_PATHS:
        _EXCLUDED_PATHS.append(path)
    return path


__all__ = [
    "C",
    "Callable",
    "Optional",
    "SimplifiedTraceback",
    "TypeVar",
    "UnfilteredStackTrace",
    "annotations",
    "api_boundary",
    "cast",
    "config",
    "dataclass",
    "filter_traceback",
    "format_exception_only",
    "functools",
    "include_frame",
    "ml_switcheroo_compiler",
    "os",
    "register_exclusion",
    "sys",
    "traceback",
    "types",
    "typing",
    "util",
]
