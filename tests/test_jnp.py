"""Tests for zero_jax.numpy (jnp) API parity."""

import numpy as np
import pytest
from zero_jax import numpy as jnp
from ml_switcheroo_compiler.tracing import _tracer, ProxyTensor


def test_unary_ops_eager():
    np.testing.assert_allclose(jnp.sin(0.0), 0.0)
    np.testing.assert_allclose(jnp.cos(0.0), 1.0)
    np.testing.assert_allclose(jnp.exp(0.0), 1.0)
    np.testing.assert_allclose(jnp.log(1.0), 0.0)


def test_transpose_eager():
    x = np.array([[1, 2], [3, 4]])
    np.testing.assert_allclose(jnp.transpose(x), [[1, 3], [2, 4]])


def test_reshape_eager():
    x = np.array([1, 2, 3, 4])
    np.testing.assert_allclose(jnp.reshape(x, (2, 2)), [[1, 2], [3, 4]])


def test_broadcast_to_eager():
    x = np.array([1, 2])
    np.testing.assert_allclose(jnp.broadcast_to(x, (2, 2)), [[1, 2], [1, 2]])


def test_concatenate_eager():
    x = np.array([1, 2])
    y = np.array([3, 4])
    np.testing.assert_allclose(jnp.concatenate([x, y]), [1, 2, 3, 4])


def test_where_eager():
    cond = np.array([True, False])
    x = np.array([1, 2])
    y = np.array([3, 4])
    np.testing.assert_allclose(jnp.where(cond, x, y), [1, 4])


def test_einsum_eager():
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    res = jnp.einsum("ij,jk->ik", x, y)
    np.testing.assert_allclose(res, np.dot(x, y))


def test_add_mul_eager():
    assert jnp.add(1, 2) == 3
    assert jnp.multiply(2, 3) == 6


def test_unary_ops_tracing():
    _tracer.start_tracing()
    x = ProxyTensor(id="x", shape=(2,))

    y = jnp.sin(x)
    assert y.shape == (2,)

    jnp.sin(1.0)  # Tracing with constant

    _tracer.stop_tracing()


def test_transpose_tracing():
    _tracer.start_tracing()
    x = ProxyTensor(id="x", shape=(2, 3))
    y = jnp.transpose(x)
    assert y.shape == (3, 2)

    z = jnp.transpose(x, axes=[1, 0])
    assert z.shape == (3, 2)

    jnp.transpose(np.array([[1, 2], [3, 4]]))
    _tracer.stop_tracing()


def test_reshape_tracing():
    _tracer.start_tracing()
    x = ProxyTensor(id="x", shape=(2, 3))
    y = jnp.reshape(x, (6,))
    assert y.shape == (6,)

    jnp.reshape(np.array([1, 2]), (2,))
    _tracer.stop_tracing()


def test_broadcast_to_tracing():
    _tracer.start_tracing()
    x = ProxyTensor(id="x", shape=(2,))
    y = jnp.broadcast_to(x, (2, 2))
    assert y.shape == (2, 2)

    jnp.broadcast_to(1.0, (2, 2))
    _tracer.stop_tracing()


def test_concatenate_tracing():
    _tracer.start_tracing()
    x = ProxyTensor(id="x", shape=(2,))
    y = ProxyTensor(id="y", shape=(3,))
    z = jnp.concatenate([x, y])
    assert z.shape == (5,)

    jnp.concatenate([x, np.array([1.0])])
    _tracer.stop_tracing()


def test_where_tracing():
    _tracer.start_tracing()
    cond = ProxyTensor(id="c", shape=(2,))
    x = ProxyTensor(id="x", shape=(2,))
    y = ProxyTensor(id="y", shape=(2,))

    z = jnp.where(cond, x, y)
    assert z.shape == (2,)

    jnp.where(True, x, 1.0)
    _tracer.stop_tracing()


def test_einsum_tracing():
    _tracer.start_tracing()
    x = ProxyTensor(id="x", shape=(2, 2))
    y = ProxyTensor(id="y", shape=(2, 2))

    jnp.einsum("ij,jk->ik", x, y)

    jnp.einsum("ij,jk->ik", x, np.array([[1, 2], [3, 4]]))
    _tracer.stop_tracing()


def test_add_mul_tracing():
    _tracer.start_tracing()
    x = ProxyTensor(id="x", shape=(2,))
    y = ProxyTensor(id="y", shape=(2,))

    jnp.add(x, y)
    jnp.add(1.0, x)

    jnp.multiply(x, y)
    jnp.multiply(x, 2.0)
    _tracer.stop_tracing()


def test_unary_op_unsupported_eager():
    from zero_jax.numpy.lax_numpy import _unary_op

    with pytest.raises(NotImplementedError):
        _unary_op(1.0, "UnsupportedOp")


def test_unary_op_transpose():
    import numpy as np
    from zero_jax.numpy.lax_numpy import _unary_op

    assert np.array_equal(
        _unary_op(np.array([[1, 2]]), "Transpose"), np.array([[1], [2]])
    )


def test_missing_numpy_methods():
    import numpy as np
    from zero_jax import numpy as jnp

    assert jnp.maximum(1, 2) == 2
    assert jnp.max([1, 2]) == 2
    assert jnp.sum([1, 2]) == 3
    assert jnp.array_equal(jnp.zeros_like(np.array([1, 2])), np.array([0, 0]))
    assert jnp.array_equal(jnp.zeros((2,)), np.array([0.0, 0.0]))
    assert jnp.abs(-1) == 1
    assert jnp.mean([1, 3]) == 2.0
    assert jnp.array_equal(jnp.array([1, 2]), np.array([1, 2]))
    assert jnp.isfinite(1)
    assert jnp.allclose(1.0, 1.0)
    assert jnp.array_equal([1], [1])
    assert jnp.broadcast_shapes((1,), (1, 2)) == (1, 2)


def test_multiply_proxy_y():
    from zero_jax import numpy as jnp
    from ml_switcheroo_compiler.tracing import _tracer, ProxyTensor

    _tracer.start_tracing()
    y = ProxyTensor(id="y", shape=(2,))
    x = 1.0
    jnp.multiply(x, y)
    _tracer.stop_tracing()


def test_lax_numpy_missing_cov():
    from zero_jax import numpy as jnp

    assert jnp.max([1, 2], where=[True, False], initial=0) == 1
    assert jnp.sum([1, 2], where=[True, False]) == 1
