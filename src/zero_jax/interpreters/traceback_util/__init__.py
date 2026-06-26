"""Frontend API routing for jax.interpreters.traceback_util."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops

C: Any = None


class SimplifiedTraceback:
    """Mock implementation for SimplifiedTraceback."""

    pass


class TypeVar:
    """Type variable."""

    pass


class UnfilteredStackTrace:
    """Mock implementation for UnfilteredStackTrace."""

    pass


annotations: Any = None


def api_boundary(*args: Any, **kwargs: Any) -> Any:
    """Wraps ``fun`` to form a boundary for filtering exception tracebacks."""
    return getattr(_ops, "api_boundary")(*args, **kwargs)


def cast(*args: Any, **kwargs: Any) -> Any:
    """Cast a value to a type."""
    return getattr(_ops, "cast")(*args, **kwargs)


from . import config


def filter_traceback(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for filter_traceback."""
    return getattr(_ops, "filter_traceback")(*args, **kwargs)


def format_exception_only(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for format_exception_only."""
    return getattr(_ops, "format_exception_only")(*args, **kwargs)


from . import functools


def include_frame(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for include_frame."""
    return getattr(_ops, "include_frame")(*args, **kwargs)


from . import os


def register_exclusion(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for register_exclusion."""
    return getattr(_ops, "register_exclusion")(*args, **kwargs)


from . import sys
from . import traceback
from . import types
from . import util
from . import xla_extension
