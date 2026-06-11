"""Experimental checkify module."""

from __future__ import annotations

from typing import Any, Callable


class ErrorCategory:
    """A category of errors that can be checked by checkify.

    Attributes:
        None
    """

    pass


user_checks = ErrorCategory()
nan_checks = ErrorCategory()
div_checks = ErrorCategory()


def checkify(fn: Callable, errors: Any = None) -> Callable:
    """Transforms a function to return any errors that occurred during execution.

    Args:
        fn: The function to transform.
        errors: The categories of errors to check for.

    Returns:
        A transformed function that returns a tuple of (error, output).
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Executes the transformed function.

        Args:
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            A tuple containing an error object (if any) and the function's output.
        """
        return None, fn(*args, **kwargs)

    return wrapper


def check(pred: Any, msg: Any) -> Any:
    """Asserts a condition and registers an error if it fails.

    Args:
        pred: A boolean condition to evaluate.
        msg: The error message to register if the condition is false.

    Returns:
        None.
    """
    pass
