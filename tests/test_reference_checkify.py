"""Reference parity tests for ``zero_jax.experimental.checkify``."""

from __future__ import annotations

import zero_jax.experimental.checkify as checkify_zero
from zero_jax._types import ArrayLikeSpec


def test_checkify_basic() -> None:
    """Verify ``zero_jax.experimental.checkify.checkify`` wraps functions and returns ``(err, out)``."""

    def f_zero(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Increment ``x`` by ``1``.

        Args:
            x: Input scalar or array.

        Returns:
            ArrayLikeSpec: Incremented value.
        """
        return x + 1

    checked_zero = checkify_zero.checkify(f_zero)
    err_z, out_z = checked_zero(5)

    assert err_z is None
    assert out_z == 6


def test_checkify_check() -> None:
    """Verify ``zero_jax.experimental.checkify.check`` executes cleanly."""
    checkify_zero.check(True, "msg")
    assert callable(checkify_zero.check)
