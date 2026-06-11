import pytest
import jax.numpy as jnp_ref
import zero_jax.numpy as jnp_zero
import numpy as np


def test_jnp_zeros(check_allclose):
    check_allclose(jnp_zero.zeros((2, 2)), jnp_ref.zeros((2, 2)))


def test_jnp_ones(check_allclose):
    check_allclose(jnp_zero.ones((2, 2)), jnp_ref.ones((2, 2)))


def test_jnp_empty(check_allclose):
    z = jnp_zero.empty((2, 2))
    r = jnp_ref.empty((2, 2))
    assert z.shape == r.shape


def test_jnp_full(check_allclose):
    check_allclose(jnp_zero.full((2, 2), 5.0), jnp_ref.full((2, 2), 5.0))


def test_jnp_zeros_like(check_allclose):
    x = np.array([1.0, 2.0])
    check_allclose(jnp_zero.zeros_like(x), jnp_ref.zeros_like(x))


def test_jnp_ones_like(check_allclose):
    x = np.array([1.0, 2.0])
    check_allclose(jnp_zero.ones_like(x), jnp_ref.ones_like(x))


def test_jnp_empty_like(check_allclose):
    x = np.array([1.0, 2.0])
    z = jnp_zero.empty_like(x)
    r = jnp_ref.empty_like(x)
    assert z.shape == r.shape


def test_jnp_full_like(check_allclose):
    x = np.array([1.0, 2.0])
    check_allclose(jnp_zero.full_like(x, 5.0), jnp_ref.full_like(x, 5.0))


def test_jnp_array(check_allclose):
    x = [1.0, 2.0]
    check_allclose(jnp_zero.array(x), jnp_ref.array(x))


def test_jnp_asarray(check_allclose):
    x = [1.0, 2.0]
    check_allclose(jnp_zero.asarray(x), jnp_ref.asarray(x))


def test_jnp_arange(check_allclose):
    check_allclose(jnp_zero.arange(5), jnp_ref.arange(5))


def test_jnp_linspace(check_allclose):
    check_allclose(jnp_zero.linspace(0, 10, 5), jnp_ref.linspace(0, 10, 5))


def test_jnp_logspace(check_allclose):
    check_allclose(jnp_zero.logspace(0, 2, 3), jnp_ref.logspace(0, 2, 3))


def test_jnp_eye(check_allclose):
    check_allclose(jnp_zero.eye(3), jnp_ref.eye(3))


def test_jnp_identity(check_allclose):
    check_allclose(jnp_zero.identity(3), jnp_ref.identity(3))


def test_jnp_meshgrid(check_allclose):
    x = np.array([1, 2])
    y = np.array([3, 4])
    Xz, Yz = jnp_zero.meshgrid(x, y)
    Xr, Yr = jnp_ref.meshgrid(x, y)
    check_allclose(Xz, Xr)
    check_allclose(Yz, Yr)


def test_jnp_sin(check_allclose):
    x = np.array([0.0, 1.0, 3.14159])
    check_allclose(jnp_zero.sin(x), jnp_ref.sin(x))


def test_jnp_cos(check_allclose):
    x = np.array([0.0, 1.0, 3.14159])
    check_allclose(jnp_zero.cos(x), jnp_ref.cos(x))


def test_jnp_exp(check_allclose):
    x = np.array([0.0, 1.0, 2.0])
    check_allclose(jnp_zero.exp(x), jnp_ref.exp(x))


def test_jnp_log(check_allclose):
    x = np.array([1.0, 2.0, 3.0])
    check_allclose(jnp_zero.log(x), jnp_ref.log(x))


def test_jnp_transpose(check_allclose):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(jnp_zero.transpose(x), jnp_ref.transpose(x))


def test_jnp_reshape(check_allclose):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    check_allclose(jnp_zero.reshape(x, (2, 2)), jnp_ref.reshape(x, (2, 2)))


def test_jnp_broadcast_to(check_allclose):
    x = np.array([1.0, 2.0])
    check_allclose(jnp_zero.broadcast_to(x, (2, 2)), jnp_ref.broadcast_to(x, (2, 2)))


def test_jnp_concatenate(check_allclose):
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    check_allclose(jnp_zero.concatenate([x, y]), jnp_ref.concatenate([x, y]))


def test_jnp_where(check_allclose):
    cond = np.array([True, False])
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    check_allclose(jnp_zero.where(cond, x, y), jnp_ref.where(cond, x, y))


def test_jnp_einsum(check_allclose):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([[5.0, 6.0], [7.0, 8.0]])
    check_allclose(
        jnp_zero.einsum("ij,jk->ik", x, y), jnp_ref.einsum("ij,jk->ik", x, y)
    )


def test_jnp_add(check_allclose):
    x, y = np.array([1.0]), np.array([2.0])
    check_allclose(jnp_zero.add(x, y), jnp_ref.add(x, y))


def test_jnp_multiply(check_allclose):
    x, y = np.array([2.0]), np.array([3.0])
    check_allclose(jnp_zero.multiply(x, y), jnp_ref.multiply(x, y))


def test_jnp_maximum(check_allclose):
    x, y = np.array([1.0, 5.0]), np.array([2.0, 3.0])
    check_allclose(jnp_zero.maximum(x, y), jnp_ref.maximum(x, y))


def test_jnp_max(check_allclose):
    x = np.array([1.0, 2.0, 3.0])
    check_allclose(jnp_zero.max(x), jnp_ref.max(x))


def test_jnp_sum(check_allclose):
    x = np.array([1.0, 2.0, 3.0])
    check_allclose(jnp_zero.sum(x), jnp_ref.sum(x))


def test_jnp_abs(check_allclose):
    x = np.array([-1.0, 2.0])
    check_allclose(jnp_zero.abs(x), jnp_ref.abs(x))


def test_jnp_mean(check_allclose):
    x = np.array([1.0, 2.0, 3.0])
    check_allclose(jnp_zero.mean(x), jnp_ref.mean(x))


def test_jnp_dot(check_allclose):
    x, y = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    check_allclose(jnp_zero.dot(x, y), jnp_ref.dot(x, y))


def test_jnp_matmul(check_allclose):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([[5.0, 6.0], [7.0, 8.0]])
    check_allclose(jnp_zero.matmul(x, y), jnp_ref.matmul(x, y))


def test_jnp_expand_dims(check_allclose):
    x = np.array([1.0, 2.0])
    check_allclose(jnp_zero.expand_dims(x, 0), jnp_ref.expand_dims(x, 0))


def test_jnp_isfinite(check_allclose):
    x = np.array([1.0, np.inf, np.nan])
    check_allclose(jnp_zero.isfinite(x), jnp_ref.isfinite(x))


def test_jnp_subtract(check_allclose):
    x, y = np.array([5.0]), np.array([2.0])
    check_allclose(jnp_zero.subtract(x, y), jnp_ref.subtract(x, y))


def test_jnp_divide(check_allclose):
    x, y = np.array([5.0]), np.array([2.0])
    check_allclose(jnp_zero.divide(x, y), jnp_ref.divide(x, y))


def test_jnp_true_divide(check_allclose):
    x, y = np.array([5.0]), np.array([2.0])
    check_allclose(jnp_zero.true_divide(x, y), jnp_ref.true_divide(x, y))


def test_jnp_floor_divide(check_allclose):
    x, y = np.array([5.0]), np.array([2.0])
    check_allclose(jnp_zero.floor_divide(x, y), jnp_ref.floor_divide(x, y))


def test_jnp_power(check_allclose):
    x, y = np.array([5.0]), np.array([2.0])
    check_allclose(jnp_zero.power(x, y), jnp_ref.power(x, y))


def test_jnp_mod(check_allclose):
    x, y = np.array([5.0]), np.array([2.0])
    check_allclose(jnp_zero.mod(x, y), jnp_ref.mod(x, y))


def test_jnp_remainder(check_allclose):
    x, y = np.array([5.0]), np.array([2.0])
    check_allclose(jnp_zero.remainder(x, y), jnp_ref.remainder(x, y))


def test_jnp_divmod(check_allclose):
    x, y = np.array([5.0]), np.array([2.0])
    # check_allclose works on tuples out of the box because of Phase 1 implementation
    check_allclose(jnp_zero.divmod(x, y), jnp_ref.divmod(x, y))


def test_jnp_negative(check_allclose):
    x = np.array([5.0, -2.0])
    check_allclose(jnp_zero.negative(x), jnp_ref.negative(x))


def test_jnp_positive(check_allclose):
    x = np.array([5.0, -2.0])
    check_allclose(jnp_zero.positive(x), jnp_ref.positive(x))


def test_jnp_sign(check_allclose):
    x = np.array([5.0, -2.0, 0.0])
    check_allclose(jnp_zero.sign(x), jnp_ref.sign(x))


def test_jnp_rint(check_allclose):
    x = np.array([5.1, -2.9, 3.5])
    check_allclose(jnp_zero.rint(x), jnp_ref.rint(x))


def test_jnp_floor(check_allclose):
    x = np.array([5.1, -2.9, 3.5])
    check_allclose(jnp_zero.floor(x), jnp_ref.floor(x))


def test_jnp_ceil(check_allclose):
    x = np.array([5.1, -2.9, 3.5])
    check_allclose(jnp_zero.ceil(x), jnp_ref.ceil(x))


def test_jnp_trunc(check_allclose):
    x = np.array([5.1, -2.9, 3.5])
    check_allclose(jnp_zero.trunc(x), jnp_ref.trunc(x))


def test_jnp_tan(check_allclose):
    x = np.array([0.0, 1.0, 3.14159 / 4])
    check_allclose(jnp_zero.tan(x), jnp_ref.tan(x))


@pytest.mark.skip(reason="Not implemented in backend")
def test_jnp_arcsin(check_allclose):
    x = np.array([0.0, 0.5, 1.0])
    check_allclose(jnp_zero.arcsin(x), jnp_ref.arcsin(x))


@pytest.mark.skip(reason="Not implemented in backend")
def test_jnp_arccos(check_allclose):
    x = np.array([0.0, 0.5, 1.0])
    check_allclose(jnp_zero.arccos(x), jnp_ref.arccos(x))


@pytest.mark.skip(reason="Not implemented in backend")
def test_jnp_arctan(check_allclose):
    x = np.array([0.0, 1.0, 5.0])
    check_allclose(jnp_zero.arctan(x), jnp_ref.arctan(x))


def test_jnp_arctan2(check_allclose):
    x, y = np.array([0.0, 1.0]), np.array([1.0, 1.0])
    check_allclose(jnp_zero.arctan2(x, y), jnp_ref.arctan2(x, y))


def test_jnp_sinh(check_allclose):
    x = np.array([0.0, 1.0, 2.0])
    check_allclose(jnp_zero.sinh(x), jnp_ref.sinh(x))


def test_jnp_cosh(check_allclose):
    x = np.array([0.0, 1.0, 2.0])
    check_allclose(jnp_zero.cosh(x), jnp_ref.cosh(x))


def test_jnp_tanh(check_allclose):
    x = np.array([0.0, 1.0, 2.0])
    check_allclose(jnp_zero.tanh(x), jnp_ref.tanh(x))


@pytest.mark.skip(reason="Not implemented in backend")
def test_jnp_arcsinh(check_allclose):
    x = np.array([0.0, 1.0, 2.0])
    check_allclose(jnp_zero.arcsinh(x), jnp_ref.arcsinh(x))


@pytest.mark.skip(reason="Not implemented in backend")
def test_jnp_arccosh(check_allclose):
    x = np.array([1.0, 2.0, 3.0])
    check_allclose(jnp_zero.arccosh(x), jnp_ref.arccosh(x))


@pytest.mark.skip(reason="Not implemented in backend")
def test_jnp_arctanh(check_allclose):
    x = np.array([0.0, 0.5, 0.9])
    check_allclose(jnp_zero.arctanh(x), jnp_ref.arctanh(x))


def test_jnp_exp2(check_allclose):
    x = np.array([0.0, 1.0, 2.0])
    check_allclose(jnp_zero.exp2(x), jnp_ref.exp2(x))


def test_jnp_expm1(check_allclose):
    x = np.array([0.0, 1.0, 2.0])
    check_allclose(jnp_zero.expm1(x), jnp_ref.expm1(x))


def test_jnp_log2(check_allclose):
    x = np.array([1.0, 2.0, 4.0])
    check_allclose(jnp_zero.log2(x), jnp_ref.log2(x))


def test_jnp_log10(check_allclose):
    x = np.array([1.0, 10.0, 100.0])
    check_allclose(jnp_zero.log10(x), jnp_ref.log10(x))


def test_jnp_log1p(check_allclose):
    x = np.array([0.0, 1.0, 2.0])
    check_allclose(jnp_zero.log1p(x), jnp_ref.log1p(x))


def test_jnp_prod(check_allclose):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(jnp_zero.prod(x), jnp_ref.prod(x))
    check_allclose(jnp_zero.prod(x, axis=0), jnp_ref.prod(x, axis=0))
    check_allclose(
        jnp_zero.prod(x, axis=1, keepdims=True), jnp_ref.prod(x, axis=1, keepdims=True)
    )


def test_jnp_var(check_allclose):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(jnp_zero.var(x), jnp_ref.var(x))
    check_allclose(jnp_zero.var(x, axis=0), jnp_ref.var(x, axis=0))
    check_allclose(
        jnp_zero.var(x, axis=1, keepdims=True), jnp_ref.var(x, axis=1, keepdims=True)
    )


def test_jnp_std(check_allclose):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(jnp_zero.std(x), jnp_ref.std(x))
    check_allclose(jnp_zero.std(x, axis=0), jnp_ref.std(x, axis=0))


def test_jnp_min(check_allclose):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(jnp_zero.min(x), jnp_ref.min(x))
    check_allclose(jnp_zero.min(x, axis=0), jnp_ref.min(x, axis=0))


def test_jnp_amin(check_allclose):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(jnp_zero.amin(x), jnp_ref.amin(x))


def test_jnp_amax(check_allclose):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(jnp_zero.amax(x), jnp_ref.amax(x))


def test_jnp_argmax(check_allclose):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(jnp_zero.argmax(x), jnp_ref.argmax(x))
    check_allclose(jnp_zero.argmax(x, axis=1), jnp_ref.argmax(x, axis=1))


def test_jnp_argmin(check_allclose):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(jnp_zero.argmin(x), jnp_ref.argmin(x))
    check_allclose(jnp_zero.argmin(x, axis=0), jnp_ref.argmin(x, axis=0))


def test_jnp_any(check_allclose):
    x = np.array([[True, False], [False, False]])
    check_allclose(jnp_zero.any(x), jnp_ref.any(x))
    check_allclose(jnp_zero.any(x, axis=0), jnp_ref.any(x, axis=0))


def test_jnp_all(check_allclose):
    x = np.array([[True, False], [True, True]])
    check_allclose(jnp_zero.all(x), jnp_ref.all(x))
    check_allclose(jnp_zero.all(x, axis=0), jnp_ref.all(x, axis=0))


def test_jnp_ravel(check_allclose):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(jnp_zero.ravel(x), jnp_ref.ravel(x))


def test_jnp_squeeze(check_allclose):
    x = np.array([[[1.0, 2.0]]])
    check_allclose(jnp_zero.squeeze(x), jnp_ref.squeeze(x))


def test_jnp_swapaxes(check_allclose):
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    check_allclose(jnp_zero.swapaxes(x, 0, 2), jnp_ref.swapaxes(x, 0, 2))


def test_jnp_moveaxis(check_allclose):
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    check_allclose(jnp_zero.moveaxis(x, 0, -1), jnp_ref.moveaxis(x, 0, -1))


def test_jnp_stack(check_allclose):
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    check_allclose(jnp_zero.stack((a, b)), jnp_ref.stack((a, b)))
    check_allclose(jnp_zero.stack((a, b), axis=1), jnp_ref.stack((a, b), axis=1))


@pytest.mark.skip(reason="Not implemented in backend")
def test_jnp_vstack(check_allclose):
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    check_allclose(jnp_zero.vstack((a, b)), jnp_ref.vstack((a, b)))


@pytest.mark.skip(reason="Not implemented in backend")
def test_jnp_hstack(check_allclose):
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    check_allclose(jnp_zero.hstack((a, b)), jnp_ref.hstack((a, b)))


@pytest.mark.skip(reason="Not implemented in backend")
def test_jnp_dstack(check_allclose):
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    check_allclose(jnp_zero.dstack((a, b)), jnp_ref.dstack((a, b)))


def test_jnp_split(check_allclose):
    x = np.arange(9.0)
    check_allclose(jnp_zero.split(x, 3), jnp_ref.split(x, 3))


@pytest.mark.skip(reason="Not implemented in backend")
def test_jnp_array_split(check_allclose):
    x = np.arange(8.0)
    check_allclose(jnp_zero.array_split(x, 3), jnp_ref.array_split(x, 3))


@pytest.mark.skip(reason="Not implemented in backend")
def test_jnp_vsplit(check_allclose):
    x = np.arange(16.0).reshape(4, 4)
    check_allclose(jnp_zero.vsplit(x, 2), jnp_ref.vsplit(x, 2))


@pytest.mark.skip(reason="Not implemented in backend")
def test_jnp_hsplit(check_allclose):
    x = np.arange(16.0).reshape(4, 4)
    check_allclose(jnp_zero.hsplit(x, 2), jnp_ref.hsplit(x, 2))


@pytest.mark.skip(reason="Not implemented in backend")
def test_jnp_dsplit(check_allclose):
    x = np.arange(16.0).reshape(2, 2, 4)
    check_allclose(jnp_zero.dsplit(x, 2), jnp_ref.dsplit(x, 2))


def test_jnp_tile(check_allclose):
    a = np.array([0, 1, 2])
    check_allclose(jnp_zero.tile(a, 2), jnp_ref.tile(a, 2))


@pytest.mark.skip(reason="Not implemented in backend")
def test_jnp_repeat(check_allclose):
    a = np.array([3, 4])
    check_allclose(jnp_zero.repeat(a, 2), jnp_ref.repeat(a, 2))


@pytest.mark.skip(reason="Not implemented in backend")
def test_jnp_pad(check_allclose):
    a = np.array([1.0, 2.0, 3.0])
    check_allclose(jnp_zero.pad(a, (1, 2)), jnp_ref.pad(a, (1, 2)))


def test_jnp_take(check_allclose):
    a = np.array([4, 3, 5, 7, 6, 8])
    indices = np.array([0, 1, 4])
    check_allclose(jnp_zero.take(a, indices), jnp_ref.take(a, indices))


@pytest.mark.skip(reason="Not implemented in backend")
def test_jnp_take_along_axis(check_allclose):
    a = np.array([[10, 30, 20], [60, 40, 50]])
    indices = np.array([[1, 0, 2], [2, 0, 1]])
    check_allclose(
        jnp_zero.take_along_axis(a, indices, axis=1),
        jnp_ref.take_along_axis(a, indices, axis=1),
    )


def test_jnp_vdot(check_allclose):
    a = np.array([1, 2, 3])
    b = np.array([0, 1, 0])
    check_allclose(jnp_zero.vdot(a, b), jnp_ref.vdot(a, b))


def test_jnp_inner(check_allclose):
    a = np.array([1, 2, 3])
    b = np.array([0, 1, 0])
    check_allclose(jnp_zero.inner(a, b), jnp_ref.inner(a, b))


def test_jnp_outer(check_allclose):
    a = np.array([1, 2, 3])
    b = np.array([0, 1, 0])
    check_allclose(jnp_zero.outer(a, b), jnp_ref.outer(a, b))


def test_jnp_tensordot(check_allclose):
    a = np.arange(60.0).reshape(3, 4, 5)
    b = np.arange(24.0).reshape(4, 3, 2)
    check_allclose(
        jnp_zero.tensordot(a, b, axes=([1, 0], [0, 1])),
        jnp_ref.tensordot(a, b, axes=([1, 0], [0, 1])),
    )
