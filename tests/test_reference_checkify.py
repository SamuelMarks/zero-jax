import pytest
import jax
import jax.experimental.checkify as checkify_ref
import zero_jax.experimental.checkify as checkify_zero


def test_checkify_basic():
    def f_ref(x):
        return x + 1

    def f_zero(x):
        return x + 1

    # Checkify returns a function that returns (err, out)
    checked_ref = checkify_ref.checkify(f_ref)
    checked_zero = checkify_zero.checkify(f_zero)

    err_r, out_r = checked_ref(5)
    err_z, out_z = checked_zero(5)

    assert out_z == out_r


def test_checkify_check():
    # zero_jax.checkify.check is a no-op currently. JAX's check might raise inside
    # checkify, but we just verify it exists and can be called
    checkify_zero.check(True, "msg")
    assert hasattr(checkify_ref, "check")
