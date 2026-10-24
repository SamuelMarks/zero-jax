"""Tests for zero_jax.experimental.checkify."""

from __future__ import annotations

import pytest

from zero_jax._types import ArrayLikeSpec
from zero_jax.experimental.checkify import (
    Error,
    ErrorCategory,
    JaxRuntimeError,
    check,
    checkify,
    div_checks,
    nan_checks,
    user_checks,
)


def test_checkify_pass_and_fail() -> None:
    """Verify checkify records failed assertions and returns None on success."""

    @checkify
    def f(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Checked function."""
        check(x > 0, "x must be positive, got {}", x)
        return x * 2

    err_ok, res_ok = f(5)
    assert err_ok is None
    assert res_ok == 10

    err_bad, res_bad = f(-3)
    assert isinstance(err_bad, Error)
    assert res_bad == -6
    assert err_bad.get() == "x must be positive, got -3"
    assert err_bad == Error(["x must be positive, got -3"])
    assert (err_bad == "invalid") is False
    assert Error([]) == None  # noqa: E711

    with pytest.raises(JaxRuntimeError, match="x must be positive, got -3"):
        err_bad.throw()

    Error([]).throw()

    # Calling check outside checkify should not crash
    check(False, "outside checkify")

    assert isinstance(user_checks, ErrorCategory)
    assert isinstance(nan_checks, ErrorCategory)
    assert isinstance(div_checks, ErrorCategory)
