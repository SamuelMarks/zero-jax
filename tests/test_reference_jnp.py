import pytest
import jax.numpy as jnp_ref
import zero_jax.numpy as jnp_zero
import numpy as np


def test_jnp_sin(check_allclose):
    x = np.array([0.0, 1.0, 3.14159])
    ref_out = jnp_ref.sin(x)
    zero_out = jnp_zero.sin(x)
    check_allclose(zero_out, ref_out)


def test_jnp_cos(check_allclose):
    x = np.array([0.0, 1.0, 3.14159])
    ref_out = jnp_ref.cos(x)
    zero_out = jnp_zero.cos(x)
    check_allclose(zero_out, ref_out)


def test_jnp_add(check_allclose):
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    ref_out = jnp_ref.add(x, y)
    zero_out = jnp_zero.add(x, y)
    check_allclose(zero_out, ref_out)


def test_jnp_reshape(check_allclose):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    ref_out = jnp_ref.reshape(x, (2, 2))
    zero_out = jnp_zero.reshape(x, (2, 2))
    check_allclose(zero_out, ref_out)
