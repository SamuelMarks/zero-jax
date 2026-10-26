"""Experimental checkify module."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec, ConfigValue, VariadicCallable

_ACTIVE_ERROR_LOG: list[list[str]] = []


class JaxRuntimeError(RuntimeError):
    """Exception raised when a ``checkify`` runtime assertion fails."""


class Error:
    """Error state object returned by a ``checkify``-transformed function."""

    def __init__(self, messages: list[str] | None = None) -> None:
        """Initialize an ``Error`` container.

        Args:
            messages: Recorded failure messages, if any.
        """
        self._messages = list(messages) if messages else []

    def get(self) -> str | None:
        """Return the first recorded error message, or ``None`` if no error occurred.

        Returns:
            str | None: Error message or ``None``.
        """
        return self._messages[0] if self._messages else None

    def throw(self) -> None:
        """Raise :class:`JaxRuntimeError` if an error was recorded during execution.

        Raises:
            JaxRuntimeError: When at least one check failed.
        """
        msg = self.get()
        if msg is not None:
            raise JaxRuntimeError(msg)

    def __eq__(self, other: ConfigValue) -> bool:
        """Allow ``err == None`` and ``Error`` equality comparisons.

        Args:
            other: Comparison value.

        Returns:
            bool: ``True`` if equivalent.
        """
        if other is None:
            return not self._messages
        if isinstance(other, Error):
            return self._messages == other._messages
        return False


class ErrorCategory:
    """A category of errors that can be checked by ``checkify``."""

    def __init__(self, category: str = "user") -> None:
        """Initialize ``ErrorCategory``.

        Args:
            category: Category name string (e.g. ``"user"``, ``"nan"``, ``"div"``).
        """
        self.category: str = str(category)
        self.name: str = str(category)


user_checks = ErrorCategory("user")
nan_checks = ErrorCategory("nan")
div_checks = ErrorCategory("div")
index_checks = ErrorCategory("index")
float_checks = ErrorCategory("float")
automatic_checks = ErrorCategory("automatic")
all_checks = ErrorCategory("all")


def checkify(
    fn: VariadicCallable[ArrayLikeSpec],
    errors: ErrorCategory | frozenset[ErrorCategory] | None = None,
) -> VariadicCallable[tuple[Error | None, ArrayLikeSpec]]:
    """Transform a function to functionalize runtime assertions and return ``(error, output)``.

    Args:
        fn: The function to transform.
        errors: Optional error category or set of categories to check.

    Returns:
        VariadicCallable[tuple[Error | None, ArrayLikeSpec]]: Transformed function returning ``(error, output)``.
    """
    _ = errors

    def wrapper(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> tuple[Error | None, ArrayLikeSpec]:
        """Execute ``fn`` while capturing any ``check`` assertion failures.

        Args:
            *args: Positional arguments to pass to ``fn``.
            **kwargs: Keyword arguments to pass to ``fn``.

        Returns:
            tuple[Error | None, ArrayLikeSpec]: ``(None, result)`` if no errors occurred, else ``(Error, result)``.
        """
        bucket: list[str] = []
        _ACTIVE_ERROR_LOG.append(bucket)
        try:
            out = fn(*args, **kwargs)
        finally:
            _ACTIVE_ERROR_LOG.pop()
        if bucket:
            return Error(bucket), out
        return None, out

    return wrapper


def check(pred: ArrayLikeSpec, msg: str = "check failed", *fmt_args: ConfigValue, **fmt_kwargs: ConfigValue) -> None:
    """Evaluate predicate ``pred`` and record an error message if ``pred`` is falsy.

    Args:
        pred: Boolean or array predicate condition to evaluate.
        msg: Error format string.
        *fmt_args: Positional format arguments for ``msg``.
        **fmt_kwargs: Keyword format arguments for ``msg``.
    """
    is_valid = bool(pred)
    if not is_valid:
        formatted = msg.format(*fmt_args, **fmt_kwargs) if (fmt_args or fmt_kwargs) else str(msg)
        if _ACTIVE_ERROR_LOG:
            _ACTIVE_ERROR_LOG[-1].append(formatted)


__all__ = [
    "Error",
    "ErrorCategory",
    "JaxRuntimeError",
    "all_checks",
    "automatic_checks",
    "check",
    "checkify",
    "div_checks",
    "float_checks",
    "index_checks",
    "nan_checks",
    "user_checks",
]
