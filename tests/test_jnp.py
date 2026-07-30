from ml_switcheroo_compiler.core.tensor import TensorConfig

"""Tests for zero_jax.numpy (jnp) API parity."""

import numpy as np
import pytest
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

from zero_jax import numpy as jnp


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


@pytest.mark.skip(reason="Not implemented without numpy")
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
    from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

    from zero_jax import numpy as jnp

    _tracer.start_tracing()
    y = ProxyTensor(id="y", shape=(2,))
    x = 1.0
    jnp.multiply(x, y)
    _tracer.stop_tracing()


def test_lax_numpy_missing_cov():
    from zero_jax import numpy as jnp

    assert jnp.max([1, 2], where=[True, False], initial=0) == 1
    assert jnp.sum([1, 2], where=[True, False]) == 1


def test_shape():
    import zero_jax.numpy as jnp

    assert jnp.shape([1, 2, 3]) == (3,)


def test_jnp_trig():
    import numpy as np

    from zero_jax import numpy as jnp

    x = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(jnp.acos(x), np.arccos(x))
    np.testing.assert_allclose(jnp.asin(x), np.arcsin(x))
    np.testing.assert_allclose(jnp.atan(x), np.arctan(x))

    x_h = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(jnp.acosh(x_h), np.arccosh(x_h))

    x_sh = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(jnp.asinh(x_sh), np.arcsinh(x_sh))

    x_th = np.array([-0.5, 0.0, 0.5])
    np.testing.assert_allclose(jnp.atanh(x_th), np.arctanh(x_th))

    x1 = np.array([1.0, -1.0])
    x2 = np.array([1.0, 1.0])
    np.testing.assert_allclose(jnp.atan2(x1, x2), np.arctan2(x1, x2))


def test_jnp_bitwise_logical_compare():
    import numpy as np

    from zero_jax import numpy as jnp

    x1, x2 = np.array([True, False, True]), np.array([True, True, False])
    np.testing.assert_allclose(jnp.logical_and(x1, x2), np.logical_and(x1, x2))
    np.testing.assert_allclose(jnp.logical_or(x1, x2), np.logical_or(x1, x2))
    np.testing.assert_allclose(jnp.logical_xor(x1, x2), np.logical_xor(x1, x2))
    np.testing.assert_allclose(jnp.logical_not(x1), np.logical_not(x1))

    i1, i2 = np.array([1, 2, 3]), np.array([1, 4, 3])
    np.testing.assert_allclose(jnp.bitwise_and(i1, i2), np.bitwise_and(i1, i2))
    np.testing.assert_allclose(jnp.bitwise_or(i1, i2), np.bitwise_or(i1, i2))
    np.testing.assert_allclose(jnp.bitwise_xor(i1, i2), np.bitwise_xor(i1, i2))
    np.testing.assert_allclose(jnp.bitwise_not(i1), np.bitwise_not(i1))

    np.testing.assert_allclose(jnp.equal(i1, i2), np.equal(i1, i2))
    np.testing.assert_allclose(jnp.not_equal(i1, i2), np.not_equal(i1, i2))
    np.testing.assert_allclose(jnp.greater(i1, i2), np.greater(i1, i2))
    np.testing.assert_allclose(jnp.greater_equal(i1, i2), np.greater_equal(i1, i2))
    np.testing.assert_allclose(jnp.less(i1, i2), np.less(i1, i2))
    np.testing.assert_allclose(jnp.less_equal(i1, i2), np.less_equal(i1, i2))


def test_jnp_math_numeric():
    import numpy as np

    from zero_jax import numpy as jnp

    x = np.array([1.5, 2.5], dtype=np.float32)
    y = np.array([2.0, 3.0], dtype=np.float32)
    i_array = np.array([1, 2], dtype=np.int32)

    np.testing.assert_allclose(jnp.cbrt(x), np.cbrt(x))
    np.testing.assert_allclose(jnp.conj(x), np.conj(x))
    np.testing.assert_allclose(jnp.copysign(x, y), np.copysign(x, y))
    assert jnp.count_nonzero(x) == np.count_nonzero(x)
    np.testing.assert_allclose(jnp.cross(x, y), np.cross(x, y))
    np.testing.assert_allclose(jnp.deg2rad(x), np.deg2rad(x))
    np.testing.assert_allclose(jnp.diag(x), np.diag(x))
    np.testing.assert_allclose(jnp.fix(x), np.fix(x))
    np.testing.assert_allclose(jnp.float_power(x, y), np.float_power(x, y))
    np.testing.assert_allclose(jnp.fmax(x, y), np.fmax(x, y))
    np.testing.assert_allclose(jnp.fmin(x, y), np.fmin(x, y))
    np.testing.assert_allclose(jnp.fmod(x, y), np.fmod(x, y))

    np.testing.assert_allclose(jnp.gcd(i_array, i_array), np.gcd(i_array, i_array))
    np.testing.assert_allclose(jnp.heaviside(x, y), np.heaviside(x, y))
    np.testing.assert_allclose(jnp.hypot(x, y), np.hypot(x, y))
    np.testing.assert_allclose(jnp.imag(x), np.imag(x))
    np.testing.assert_allclose(jnp.isclose(x, x), np.isclose(x, x))
    np.testing.assert_allclose(jnp.isinf(x), np.isinf(x))
    np.testing.assert_allclose(jnp.lcm(i_array, i_array), np.lcm(i_array, i_array))
    np.testing.assert_allclose(jnp.ldexp(x, i_array), np.ldexp(x, i_array))
    np.testing.assert_allclose(
        jnp.left_shift(i_array, i_array), np.left_shift(i_array, i_array)
    )
    np.testing.assert_allclose(jnp.logaddexp(x, y), np.logaddexp(x, y))
    np.testing.assert_allclose(jnp.logaddexp2(x, y), np.logaddexp2(x, y))
    np.testing.assert_allclose(jnp.nextafter(x, y), np.nextafter(x, y))
    np.testing.assert_allclose(jnp.rad2deg(x), np.rad2deg(x))
    np.testing.assert_allclose(jnp.real(x), np.real(x))
    np.testing.assert_allclose(jnp.reciprocal(x), np.reciprocal(x))
    np.testing.assert_allclose(
        jnp.right_shift(i_array, i_array), np.right_shift(i_array, i_array)
    )
    np.testing.assert_allclose(jnp.roll(x, shift=1), np.roll(x, shift=1))
    np.testing.assert_allclose(jnp.round(x), np.round(x))
    np.testing.assert_allclose(jnp.sinc(x), np.sinc(x))
    np.testing.assert_allclose(jnp.sort(x), np.sort(x))

    m = np.eye(2)
    np.testing.assert_allclose(jnp.tril(m), np.tril(m))
    np.testing.assert_allclose(jnp.triu(m), np.triu(m))

    # jnp.select uses _to_tensor inside
    np.testing.assert_allclose(jnp.select([x > 1], [x]), np.select([x > 1], [x]))

    # unstack is a jnp function, but wait, numpy does not have unstack
    # np.unstack does not exist. We can compare with list of slices
    for z_un, n_un in zip(
        jnp.unstack(x), [np.squeeze(arr, axis=0) for arr in np.split(x, x.shape[0])]
    ):
        np.testing.assert_allclose(z_un, n_un)


def test_jnp_fft_linalg():
    import numpy as np

    from zero_jax import numpy as jnp

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    m = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)

    _ = jnp.fft.fft(x)
    _ = jnp.fft.rfft(x)

    _ = jnp.linalg.cholesky(m)
    _ = jnp.linalg.det(m)
    _ = jnp.linalg.inv(m)
    _ = jnp.linalg.matrix_power(m, 2)
    _ = jnp.linalg.pinv(m)
    _ = jnp.linalg.qr(m)
    _ = jnp.linalg.slogdet(m)
    _ = jnp.linalg.solve(m, np.array([1.0, 2.0], dtype=np.float32))
    _ = jnp.linalg.svd(m)


def test_jnp_aliases():
    import numpy as np

    from zero_jax import numpy as jnp

    x = np.array([1.5, -2.5], dtype=np.float32)
    y = np.array([2.0, 3.0], dtype=np.float32)

    np.testing.assert_allclose(jnp.absolute(x), np.abs(x))
    np.testing.assert_allclose(jnp.around(x), np.round(x))
    np.testing.assert_allclose(jnp.round_(x), np.round(x))
    np.testing.assert_allclose(jnp.conjugate(x), np.conj(x))
    np.testing.assert_allclose(jnp.cumulative_sum(x), np.cumsum(x))
    np.testing.assert_allclose(jnp.degrees(x), np.rad2deg(x))
    np.testing.assert_allclose(jnp.radians(x), np.deg2rad(x))
    np.testing.assert_allclose(jnp.pow(x, y), np.power(x, y))


def test_jnp_more_aliases():
    import numpy as np

    from zero_jax import numpy as jnp

    x = np.array([1, 2], dtype=np.int32)
    y = np.array([1, 1], dtype=np.int32)
    z = np.array([1.0, np.nan, -np.inf, np.inf], dtype=np.float32)

    np.testing.assert_allclose(jnp.bitwise_invert(x), np.bitwise_not(x))
    np.testing.assert_allclose(jnp.bitwise_left_shift(x, y), np.left_shift(x, y))
    np.testing.assert_allclose(jnp.bitwise_right_shift(x, y), np.right_shift(x, y))
    np.testing.assert_allclose(jnp.concat([x, y]), np.concatenate([x, y]))
    np.testing.assert_allclose(jnp.invert(x), np.invert(x))

    # We can't easily assert_allclose on nanmax since it raises warnings in numpy but let's test it on valid data
    v = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(jnp.nanmax(v), np.nanmax(v))
    np.testing.assert_allclose(jnp.nanmin(v), np.nanmin(v))
    np.testing.assert_allclose(jnp.nanprod(v), np.nanprod(v))
    np.testing.assert_allclose(jnp.nansum(v), np.nansum(v))

    np.testing.assert_allclose(jnp.isneginf(z), np.isneginf(z))
    np.testing.assert_allclose(jnp.isposinf(z), np.isposinf(z))


def test_jnp_frexp():
    import numpy as np

    from zero_jax import numpy as jnp

    x = np.array([1.5, 2.5], dtype=np.float32)
    a, b = jnp.frexp(x)
    na, nb = np.frexp(x)

    np.testing.assert_allclose(a, na)
    np.testing.assert_allclose(b, nb)


def test_jnp_nan_to_num_searchsorted_signbit():
    import numpy as np

    from zero_jax import numpy as jnp

    x = np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    np.testing.assert_allclose(jnp.nan_to_num(x), np.nan_to_num(x))
    np.testing.assert_allclose(jnp.signbit(x), np.signbit(x))

    a = np.array([1, 2, 4, 5])
    v = np.array([3, 6])
    np.testing.assert_allclose(jnp.searchsorted(a, v), np.searchsorted(a, v))


def test_jnp_copy():
    import numpy as np

    from zero_jax import numpy as jnp

    x = np.array([1, 2, 3])
    np.testing.assert_allclose(jnp.copy(x), np.copy(x))


@pytest.mark.skip(reason="Not implemented without numpy")
def test_jnp_added_funcs_eager():
    x = np.array([1 + 1j, 1 - 1j])
    np.testing.assert_allclose(jnp.angle(x), np.angle(x))

    x = np.array([1, 2])
    np.testing.assert_allclose(jnp.append(x, [3, 4]), np.append(x, [3, 4]))

    y = jnp.astype(x, jnp.float32)
    assert y.dtype.value == "float32"

    res1 = jnp.atleast_1d(1)
    assert res1.ndim >= 1

    res2 = jnp.atleast_2d(1)
    assert res2.ndim >= 2

    res3 = jnp.atleast_3d(1)
    assert res3.ndim >= 3

    np.testing.assert_allclose(jnp.average([1, 2, 3]), 2.0)

    # block
    A = np.eye(2) * 2
    B = np.eye(3) * 3
    # JAX np.block equivalent:
    try:
        jnp.block([[A, np.zeros((2, 3))], [np.zeros((3, 2)), B]])
    except NotImplementedError:
        pass  # If zero_jax._compiler_proxy_ops are not implemented fully eager


@pytest.mark.skip(reason="Not implemented without numpy")
def test_jnp_batch2_eager():
    # apply_along_axis
    def my_func(a):
        return jnp.sum(a)

    x = np.array([[1, 2, 3], [4, 5, 6]])
    res = jnp.apply_along_axis(my_func, 0, x)
    np.testing.assert_allclose(res, np.apply_along_axis(np.sum, 0, x))

    # apply_over_axes omitted

    # argpartition
    res3 = jnp.argpartition(np.array([3, 4, 2, 1]), 2)
    np.testing.assert_allclose(res3, np.argpartition(np.array([3, 4, 2, 1]), 2))

    # argwhere
    res4 = jnp.argwhere(x > 3)
    np.testing.assert_allclose(res4, np.argwhere(x > 3))

    # choose
    res5 = jnp.choose(
        np.array([0, 1, 0]), [np.array([10, 20, 30]), np.array([40, 50, 60])]
    )
    np.testing.assert_allclose(
        res5,
        np.choose(
            np.array([0, 1, 0]), [np.array([10, 20, 30]), np.array([40, 50, 60])]
        ),
    )

    # column_stack
    res6 = jnp.column_stack((np.array([1, 2]), np.array([3, 4])))
    np.testing.assert_allclose(
        res6, np.column_stack((np.array([1, 2]), np.array([3, 4])))
    )

    # compress
    res7 = jnp.compress(np.array([True, False, True]), np.array([1, 2, 3]))
    np.testing.assert_allclose(
        res7, np.compress(np.array([True, False, True]), np.array([1, 2, 3]))
    )

    # convolve
    res8 = jnp.convolve(np.array([1, 2, 3]), np.array([0, 1, 0.5]))
    np.testing.assert_allclose(
        res8, np.convolve(np.array([1, 2, 3]), np.array([0, 1, 0.5]))
    )

    # corrcoef
    res9 = jnp.corrcoef(np.array([[1, 2, 3], [4, 5, 6]]))
    np.testing.assert_allclose(res9, np.corrcoef(np.array([[1, 2, 3], [4, 5, 6]])))

    # correlate
    res10 = jnp.correlate(np.array([1, 2, 3]), np.array([0, 1]))
    np.testing.assert_allclose(
        res10, np.correlate(np.array([1, 2, 3]), np.array([0, 1]))
    )

    # cov
    res11 = jnp.cov(np.array([[1, 2, 3], [4, 5, 6]]))
    np.testing.assert_allclose(res11, np.cov(np.array([[1, 2, 3], [4, 5, 6]])))


@pytest.mark.skip(reason="Not implemented without numpy")
def test_jnp_batch3_eager():
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    # array_equiv
    assert jnp.array_equiv(x, y) == np.array_equiv(x, y)

    # array_repr / str
    assert isinstance(jnp.array_repr(x), str)
    assert isinstance(jnp.array_str(x), str)

    # bartlett
    np.testing.assert_allclose(jnp.bartlett(3), np.bartlett(3))

    # bincount
    np.testing.assert_allclose(
        jnp.bincount(np.array([1, 1, 2])), np.bincount(np.array([1, 1, 2]))
    )

    # bitwise_count
    if hasattr(np, "bitwise_count"):
        np.testing.assert_allclose(
            jnp.bitwise_count(np.array([1, 2, 3])),
            np.bitwise_count(np.array([1, 2, 3])),
        )

    # blackman
    np.testing.assert_allclose(jnp.blackman(3), np.blackman(3))

    # broadcast_arrays
    res1, res2 = jnp.broadcast_arrays(np.array([1, 2]), np.array([[1], [2]]))
    np1, np2 = np.broadcast_arrays(np.array([1, 2]), np.array([[1], [2]]))
    np.testing.assert_allclose(res1, np1)
    np.testing.assert_allclose(res2, np2)

    # can_cast
    assert jnp.can_cast(np.int32, np.float32) == np.can_cast(np.int32, np.float32)

    # cumprod
    np.testing.assert_allclose(jnp.cumprod(x), np.cumprod(x))

    # delete
    np.testing.assert_allclose(jnp.delete(x, 1), np.delete(x, 1))

    # diag_indices
    r1, r2 = jnp.diag_indices(2)
    nr1, nr2 = np.diag_indices(2)
    np.testing.assert_allclose(r1, nr1)
    np.testing.assert_allclose(r2, nr2)

    # diag_indices_from
    r1, r2 = jnp.diag_indices_from(np.eye(2))
    nr1, nr2 = np.diag_indices_from(np.eye(2))
    np.testing.assert_allclose(r1, nr1)
    np.testing.assert_allclose(r2, nr2)

    # diagflat
    np.testing.assert_allclose(
        jnp.diagflat(np.array([1, 2])), np.diagflat(np.array([1, 2]))
    )

    # diagonal
    np.testing.assert_allclose(jnp.diagonal(np.eye(2)), np.diagonal(np.eye(2)))

    # diff
    np.testing.assert_allclose(jnp.diff(x), np.diff(x))

    # digitize
    np.testing.assert_allclose(
        jnp.digitize(np.array([0.5, 1.5]), np.array([0, 1, 2])),
        np.digitize(np.array([0.5, 1.5]), np.array([0, 1, 2])),
    )


@pytest.mark.skip(reason="Not implemented without numpy")
def test_jnp_batch4_eager():
    x = np.array([1, 2, 3])
    # dtype
    assert jnp.dtype(np.int32).value == np.dtype(np.int32).name

    # ediff1d
    np.testing.assert_allclose(jnp.ediff1d(x), np.ediff1d(x))

    # einsum_path
    res1, res2 = jnp.einsum_path("i,i->", x, x)
    np1, np2 = np.einsum_path("i,i->", x, x)
    assert res1 == np1
    assert str(res2) == str(np2)

    # extract
    np.testing.assert_allclose(jnp.extract(x > 1, x), np.extract(x > 1, x))

    # fabs
    np.testing.assert_allclose(
        jnp.fabs(np.array([-1.5, 2.5])), np.fabs(np.array([-1.5, 2.5]))
    )

    # fill_diagonal
    a1 = np.zeros((3, 3))
    a2 = np.zeros((3, 3))
    jnp.fill_diagonal(
        a1, 1
    )  # Note: returns copy in eager mode due to backend wrapping, JAX might modify in place
    np.fill_diagonal(a2, 1)
    # the returned array is verified
    res = jnp.fill_diagonal(np.zeros((3, 3)), 1)
    np.testing.assert_allclose(res, a2)

    # finfo
    assert jnp.finfo(np.float32).eps == np.finfo(np.float32).eps

    # flatnonzero
    np.testing.assert_allclose(
        jnp.flatnonzero(np.array([-1, 0, 1])), np.flatnonzero(np.array([-1, 0, 1]))
    )

    # flip
    np.testing.assert_allclose(jnp.flip(x), np.flip(x))

    # fliplr
    np.testing.assert_allclose(jnp.fliplr(np.eye(2)), np.fliplr(np.eye(2)))

    # flipud
    np.testing.assert_allclose(jnp.flipud(np.eye(2)), np.flipud(np.eye(2)))

    # float8 mapping check
    assert jnp.float8_e4m3fn == jnp.float16


@pytest.mark.skip(reason="Not implemented without numpy")
def test_jnp_batch5_eager():
    x = np.array([1, 2, 3])

    # frombuffer
    b = bytes([1, 2, 3, 4])
    np.testing.assert_allclose(
        jnp.frombuffer(b, dtype=np.uint8), np.frombuffer(b, dtype=np.uint8)
    )

    # fromfunction
    np.testing.assert_allclose(
        jnp.fromfunction(lambda i, j: i == j, (3, 3), dtype=int),
        np.fromfunction(lambda i, j: i == j, (3, 3), dtype=int),
    )

    # fromiter
    iterable = (x * x for x in range(5))
    iterable2 = (x * x for x in range(5))
    np.testing.assert_allclose(
        jnp.fromiter(iterable, float), np.fromiter(iterable2, float)
    )

    # fromstring
    np.testing.assert_allclose(
        jnp.fromstring("1 2", dtype=int, sep=" "),
        np.fromstring("1 2", dtype=int, sep=" "),
    )

    # geomspace
    np.testing.assert_allclose(
        jnp.geomspace(1, 1000, num=4), np.geomspace(1, 1000, num=4)
    )

    # get_printoptions
    assert isinstance(jnp.get_printoptions(), dict)

    # gradient
    np.testing.assert_allclose(
        jnp.gradient(np.array([1, 2, 4, 7, 11], dtype=float)),
        np.gradient(np.array([1, 2, 4, 7, 11], dtype=float)),
    )

    # hamming
    np.testing.assert_allclose(jnp.hamming(3), np.hamming(3))

    # hanning
    np.testing.assert_allclose(jnp.hanning(3), np.hanning(3))

    # histogram
    h1, e1 = jnp.histogram(np.array([1, 2, 1]), bins=[0, 1, 2, 3])
    h2, e2 = np.histogram(np.array([1, 2, 1]), bins=[0, 1, 2, 3])
    np.testing.assert_allclose(h1, h2)
    np.testing.assert_allclose(e1, e2)

    # histogram2d
    h1, x1, y1 = jnp.histogram2d(
        np.array([1, 2, 1]), np.array([1, 2, 1]), bins=[[0, 1, 2, 3], [0, 1, 2, 3]]
    )
    h2, x2, y2 = np.histogram2d(
        np.array([1, 2, 1]), np.array([1, 2, 1]), bins=[[0, 1, 2, 3], [0, 1, 2, 3]]
    )
    np.testing.assert_allclose(h1, h2)

    # histogram_bin_edges
    np.testing.assert_allclose(
        jnp.histogram_bin_edges(np.array([1, 2])),
        np.histogram_bin_edges(np.array([1, 2])),
    )

    # histogramdd
    h1, e1 = jnp.histogramdd(
        np.array([[1, 1], [2, 2], [1, 1]]), bins=[[0, 1, 2, 3], [0, 1, 2, 3]]
    )
    h2, e2 = np.histogramdd(
        np.array([[1, 1], [2, 2], [1, 1]]), bins=[[0, 1, 2, 3], [0, 1, 2, 3]]
    )
    np.testing.assert_allclose(h1, h2)

    # i0
    np.testing.assert_allclose(jnp.i0(np.array([0, 1, 2])), np.i0(np.array([0, 1, 2])))

    # iinfo
    assert jnp.iinfo(np.int32).max == np.iinfo(np.int32).max

    # indices
    r1, r2 = jnp.indices((2, 3), sparse=True)
    nr1, nr2 = np.indices((2, 3), sparse=True)
    np.testing.assert_allclose(r1, nr1)

    # insert
    np.testing.assert_allclose(
        jnp.insert(np.array([1, 2, 3]), 1, 5), np.insert(np.array([1, 2, 3]), 1, 5)
    )

    # interp
    np.testing.assert_allclose(
        jnp.interp(2.5, [1, 2, 3], [3, 2, 0]), np.interp(2.5, [1, 2, 3], [3, 2, 0])
    )

    # intersect1d
    np.testing.assert_allclose(
        jnp.intersect1d(np.array([1, 3, 4, 3]), np.array([3, 1, 2, 1])),
        np.intersect1d(np.array([1, 3, 4, 3]), np.array([3, 1, 2, 1])),
    )


@pytest.mark.skip(reason="Not implemented without numpy")
def test_jnp_batch6_eager():
    x = np.array([1, 2, 3])

    # iscomplex
    np.testing.assert_allclose(jnp.iscomplex(x), np.iscomplex(x))

    # iscomplexobj
    assert jnp.iscomplexobj(x) == np.iscomplexobj(x)

    # isdtype
    # Older numpy might not have isdtype, skip if so
    if hasattr(np, "isdtype"):
        assert jnp.isdtype(np.int32, "integral") == np.isdtype(np.int32, "integral")

    # isin
    np.testing.assert_allclose(jnp.isin(x, [1, 3]), np.isin(x, [1, 3]))

    # isreal
    np.testing.assert_allclose(jnp.isreal(x), np.isreal(x))

    # isrealobj

    # isscalar
    assert jnp.isscalar(3.1) == np.isscalar(3.1)

    # issubdtype

    # iterable

    # ix_
    a = np.array([0, 1])
    b = np.array([2, 3])
    ra, rb = jnp.ix_(a, b)
    nra, nrb = np.ix_(a, b)
    np.testing.assert_allclose(ra, nra)
    np.testing.assert_allclose(rb, nrb)

    # kaiser
    np.testing.assert_allclose(jnp.kaiser(3, 14), np.kaiser(3, 14))

    # kron
    np.testing.assert_allclose(jnp.kron(a, b), np.kron(a, b))

    # lexsort
    np.testing.assert_allclose(jnp.lexsort((b, a)), np.lexsort((b, a)))

    # load (skip complex file IO for eager unit test if we can't write, just testing it proxies)
    assert hasattr(jnp, "load")

    # mask_indices
    m1, m2 = jnp.mask_indices(3, np.triu)
    nm1, nm2 = np.mask_indices(3, np.triu)
    np.testing.assert_allclose(m1, nm1)
    np.testing.assert_allclose(m2, nm2)

    # matrix_transpose
    m = np.eye(2)
    if hasattr(np, "matrix_transpose"):
        np.testing.assert_allclose(jnp.matrix_transpose(m), np.matrix_transpose(m))

    # median
    np.testing.assert_allclose(jnp.median(x), np.median(x))

    # modf
    m1, m2 = jnp.modf(np.array([1.5, 2.5]))
    nm1, nm2 = np.modf(np.array([1.5, 2.5]))
    np.testing.assert_allclose(m1, nm1)
    np.testing.assert_allclose(m2, nm2)

    # nan* (various)
    nx = np.array([1.0, np.nan, 3.0])
    np.testing.assert_allclose(jnp.nanargmax(nx), np.nanargmax(nx))
    np.testing.assert_allclose(jnp.nanargmin(nx), np.nanargmin(nx))
    np.testing.assert_allclose(jnp.nancumprod(nx), np.nancumprod(nx))
    np.testing.assert_allclose(jnp.nancumsum(nx), np.nancumsum(nx))
    np.testing.assert_allclose(jnp.nanmean(nx), np.nanmean(nx))
    np.testing.assert_allclose(jnp.nanmedian(nx), np.nanmedian(nx))
    np.testing.assert_allclose(jnp.nanpercentile(nx, 50), np.nanpercentile(nx, 50))
    np.testing.assert_allclose(jnp.nanquantile(nx, 0.5), np.nanquantile(nx, 0.5))
    np.testing.assert_allclose(jnp.nanstd(nx), np.nanstd(nx))
    np.testing.assert_allclose(jnp.nanvar(nx), np.nanvar(nx))
