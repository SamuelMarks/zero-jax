"""Tests for zero_jax.lax primitives."""

import pytest
from zero_jax.lax import cond, scan
from ml_switcheroo_compiler.tracing import _tracer, ProxyTensor


def test_cond_eager():
    assert cond(True, lambda x: x + 1, lambda x: x - 1, 5) == 6
    assert cond(False, lambda x: x + 1, lambda x: x - 1, 5) == 4


def test_scan_eager():
    def f(carry, x):
        return carry + 1, carry * 2

    carry, ys = scan(f, 0, [1, 2, 3])
    assert carry == 3
    assert ys == [0, 2, 4]


def test_scan_eager_no_xs():
    def f(carry, x):
        return carry + 1, carry * 2

    carry, ys = scan(f, 0, None, length=3)
    assert carry == 3
    assert ys == [0, 2, 4]

    with pytest.raises(ValueError):
        scan(f, 0, None)


def test_cond_tracing():
    _tracer.start_tracing()
    pred = ProxyTensor(id="p", shape=())
    x = ProxyTensor(id="x", shape=())

    res = cond(pred, lambda x: x, lambda x: x, x)
    assert res.shape == ()
    _tracer.stop_tracing()


def test_scan_tracing():
    _tracer.start_tracing()
    init = ProxyTensor(id="i", shape=())
    xs = ProxyTensor(id="xs", shape=(10,))

    carry, ys = scan(lambda c, x: (c, x), init, xs)
    assert carry.shape == ()
    assert ys.shape == ()
    _tracer.stop_tracing()


def test_stop_gradient():
    from zero_jax.lax import stop_gradient

    assert stop_gradient(5) == 5
