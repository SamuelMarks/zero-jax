import pytest

from zero_jax.lax import primitives as lax
from zero_jax.numpy import array


def add(x, y):
    return x + y


def mul(x, y):
    return x * y


def max(x, y):
    return x if x > y else y


def min(x, y):
    return x if x < y else y


def test_lax_reduce_others():
    x = array([1, 2, 3])

    res1 = lax.reduce(x, 1, mul, (0,))
    res2 = lax.reduce(x, 0, max, (0,))
    res3 = lax.reduce(x, 10, min, (0,))
    res4 = lax.reduce(x, 0, lambda a, b: a, (0,))

    assert res1 is not None
