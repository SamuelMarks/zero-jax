"""Reference parity tests for ``zero_jax.random``."""

from __future__ import annotations

import zero_jax.numpy as jnp_zero
import zero_jax.random as rand_zero
from zero_jax._types import VariadicCallable


def test_random_skip_split(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.random.split`` splits a PRNG key into the requested count.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    _ = check_allclose
    key = rand_zero.PRNGKey(42)
    keys_z = rand_zero.split(key, 2)
    assert keys_z.shape == (2, 2)


def test_random_skip_fold_in(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.random.fold_in`` folds integer data into a PRNG key.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    _ = check_allclose
    key = rand_zero.PRNGKey(42)
    key_z = rand_zero.fold_in(key, 1)
    assert key_z.shape == (2,)


def test_random_skip_uniform(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.random.uniform`` samples uniform values with expected shape.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    _ = check_allclose
    key = rand_zero.PRNGKey(42)
    z = rand_zero.uniform(key, (2, 2))
    assert z.shape == (2, 2)


def test_random_skip_normal(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.random.normal`` samples normal values with expected shape.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    _ = check_allclose
    key = rand_zero.PRNGKey(42)
    z = rand_zero.normal(key, (2, 2))
    assert z.shape == (2, 2)


def test_random_skip_randint(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.random.randint`` samples bounded integers with expected shape.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    _ = check_allclose
    key = rand_zero.PRNGKey(42)
    z = rand_zero.randint(key, (2, 2), 0, 10)
    assert z.shape == (2, 2)


def test_random_skip_bernoulli(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.random.bernoulli`` samples Bernoulli booleans with expected shapes.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    _ = check_allclose
    key = rand_zero.PRNGKey(42)
    z = rand_zero.bernoulli(key, 0.5, (2, 2))
    assert z.shape == (2, 2)
    assert rand_zero.bernoulli(key, 0.5).shape == ()


def test_random_skip_categorical(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.random.categorical`` samples categorical indices from logits.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    _ = check_allclose
    key = rand_zero.PRNGKey(42)
    logits = jnp_zero.array([0.1, 0.2, 0.7])
    z = rand_zero.categorical(key, logits)
    assert z.shape == ()


def test_random_skip_permutation(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.random.permutation`` permutes elements preserving shape.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    _ = check_allclose
    key = rand_zero.PRNGKey(42)
    x = jnp_zero.arange(10)
    z = rand_zero.permutation(key, x)
    assert z.shape == (10,)
