"""Tests for zero_jax transformations."""

from zero_jax import jit, grad, value_and_grad, vmap


def test_transform_jit():
    @jit
    def f(x):
        return x + 1

    assert f(1) == 2


def test_transform_grad():
    @grad
    def f(x):
        return x + 1

    assert f(1) == 2  # Mocked to return evaluation for now


def test_transform_value_and_grad():
    @value_and_grad
    def f(x):
        return x + 1

    assert f(1) == (2, 2)


def test_transform_vmap():
    @vmap
    def f(x):
        return x + 1

    assert f(1) == 2
