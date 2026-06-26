"""Frontend API routing for jax.interpreters.traceback_util."""

from typing import Any

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
    raise NotImplementedError("api_boundary not yet implemented in zero-jax")


def cast(*args: Any, **kwargs: Any) -> Any:
    """Cast a value to a type."""
    raise NotImplementedError("cast not yet implemented in zero-jax")


from . import config


def filter_traceback(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for filter_traceback."""
    raise NotImplementedError("filter_traceback not yet implemented in zero-jax")


def format_exception_only(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for format_exception_only."""
    raise NotImplementedError("format_exception_only not yet implemented in zero-jax")


from . import functools


def include_frame(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for include_frame."""
    raise NotImplementedError("include_frame not yet implemented in zero-jax")


from . import os


def register_exclusion(*args: Any, **kwargs: Any) -> Any:
    """Mock implementation for register_exclusion."""
    raise NotImplementedError("register_exclusion not yet implemented in zero-jax")


from . import sys
from . import traceback
from . import types
from . import util
from . import xla_extension
