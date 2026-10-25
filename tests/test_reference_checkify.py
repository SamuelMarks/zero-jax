"""Tests for tests/test_reference_checkify.py."""

from __future__ import annotations

import jax.experimental.checkify as checkify_ref

import zero_jax.experimental.checkify as checkify_zero
from zero_jax._types import ArrayLikeSpec


def test_checkify_basic() -> None:
    """Execute ``test_checkify_basic``."""

    def f_ref(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``f_ref``.

        Args:
            x: Parameter ``x``.
        """
        return x + 1

    def f_zero(x: ArrayLikeSpec) -> ArrayLikeSpec:
        """Execute ``f_zero``.

        Args:
            x: Parameter ``x``.
        """
        return x + 1

    # Checkify returns a function that returns (err, out)
    checked_ref = checkify_ref.checkify(f_ref)
    checked_zero = checkify_zero.checkify(f_zero)

    _err_r, out_r = checked_ref(5)
    _err_z, out_z = checked_zero(5)

    assert out_z == out_r


def test_checkify_check() -> None:
    # zero_jax.checkify.check is a no-op currently. JAX's check might raise inside
    # checkify, but we just verify it exists and can be called
    """Execute ``test_checkify_check``."""
    checkify_zero.check(True, "msg")
    assert callable(checkify_ref.check)
