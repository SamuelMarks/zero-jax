"""Module docstring."""

from typing import Any


class ErrorCategory:
    """ErrorCategory class."""

    pass


user_checks = ErrorCategory()
nan_checks = ErrorCategory()
div_checks = ErrorCategory()


def checkify(fn: Any, errors: Any = None) -> Any:
    """Checkify function."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper function."""
        return None, fn(*args, **kwargs)

    return wrapper


def check(pred: Any, msg: Any) -> Any:
    """Check function."""
    pass
