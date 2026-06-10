class ErrorCategory:
    pass


user_checks = ErrorCategory()
nan_checks = ErrorCategory()
div_checks = ErrorCategory()


def checkify(fn, errors=None):
    def wrapper(*args, **kwargs):
        return None, fn(*args, **kwargs)

    return wrapper
