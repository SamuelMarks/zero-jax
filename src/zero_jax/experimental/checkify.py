"""Module docstring."""


class ErrorCategory:
    """Class docstring."""

    pass


user_checks = ErrorCategory()
nan_checks = ErrorCategory()
div_checks = ErrorCategory()


def checkify(fn, errors=None):
    """Function docstring."""
    return lambda *a, **k: (None, fn(*a, **k))
