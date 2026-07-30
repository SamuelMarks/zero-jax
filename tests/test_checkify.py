from zero_jax.experimental.checkify import (
    ErrorCategory,
    checkify,
    div_checks,
    nan_checks,
    user_checks,
)


def test_checkify():
    @checkify
    def f(x):
        return x

    err, res = f(10)
    assert err is None
    assert res == 10

    assert isinstance(user_checks, ErrorCategory)
    assert isinstance(nan_checks, ErrorCategory)
    assert isinstance(div_checks, ErrorCategory)
