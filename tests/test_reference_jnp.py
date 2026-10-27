"""Reference parity tests for ``zero_jax.numpy`` against analytical math references."""

from __future__ import annotations

import math

import zero_jax.numpy as jnp_zero
from zero_jax._types import VariadicCallable


def test_jnp_zeros(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.zeros`` matches expected zero array.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    check_allclose(jnp_zero.zeros((2, 2)), jnp_zero.array([[0.0, 0.0], [0.0, 0.0]]))


def test_jnp_ones(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.ones`` matches expected ones array.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    check_allclose(jnp_zero.ones((2, 2)), jnp_zero.array([[1.0, 1.0], [1.0, 1.0]]))


def test_jnp_empty(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.empty`` returns the requested shape.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    _ = check_allclose
    z = jnp_zero.empty((2, 2))
    assert z.shape == (2, 2)


def test_jnp_full(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.full`` matches expected constant array.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    check_allclose(jnp_zero.full((2, 2), 5.0), jnp_zero.array([[5.0, 5.0], [5.0, 5.0]]))


def test_jnp_zeros_like(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.zeros_like`` matches expected zero array.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([1.0, 2.0])
    check_allclose(jnp_zero.zeros_like(x), jnp_zero.array([0.0, 0.0]))


def test_jnp_ones_like(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.ones_like`` matches expected ones array.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([1.0, 2.0])
    check_allclose(jnp_zero.ones_like(x), jnp_zero.array([1.0, 1.0]))


def test_jnp_empty_like(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.empty_like`` preserves input shape.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    _ = check_allclose
    x = jnp_zero.array([1.0, 2.0])
    z = jnp_zero.empty_like(x)
    assert z.shape == (2,)


def test_jnp_full_like(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.full_like`` fills with the specified scalar.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([1.0, 2.0])
    check_allclose(jnp_zero.full_like(x, 5.0), jnp_zero.array([5.0, 5.0]))


def test_jnp_array(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.array`` constructs an array from a list.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = [1.0, 2.0]
    check_allclose(jnp_zero.array(x), [1.0, 2.0])


def test_jnp_asarray(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.asarray`` converts a list to an array.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = [1.0, 2.0]
    check_allclose(jnp_zero.asarray(x), [1.0, 2.0])


def test_jnp_arange(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.arange`` produces an integer range.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    check_allclose(jnp_zero.arange(5), jnp_zero.array([0, 1, 2, 3, 4]))


def test_jnp_linspace(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.linspace`` produces linearly spaced samples.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    check_allclose(jnp_zero.linspace(0, 10, 5), jnp_zero.array([0.0, 2.5, 5.0, 7.5, 10.0]))


def test_jnp_logspace(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.logspace`` produces logarithmically spaced samples.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    check_allclose(jnp_zero.logspace(0, 2, 3), jnp_zero.array([1.0, 10.0, 100.0]))


def test_jnp_eye(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.eye`` produces a 2D identity matrix.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    check_allclose(
        jnp_zero.eye(3),
        jnp_zero.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )


def test_jnp_identity(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.identity`` produces a square identity matrix.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    check_allclose(
        jnp_zero.identity(3),
        jnp_zero.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )


def test_jnp_meshgrid(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.meshgrid`` produces coordinate matrices.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([1, 2])
    y = jnp_zero.array([3, 4])
    Xz, Yz = jnp_zero.meshgrid(x, y)
    Xr = jnp_zero.array([[1, 2], [1, 2]])
    Yr = jnp_zero.array([[3, 3], [4, 4]])
    check_allclose(Xz, Xr)
    check_allclose(Yz, Yr)


def test_jnp_sin(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.sin`` against ``math.sin``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [0.0, 1.0, 3.14159]
    x = jnp_zero.array(vals)
    check_allclose(jnp_zero.sin(x), jnp_zero.array([math.sin(v) for v in vals]))


def test_jnp_cos(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.cos`` against ``math.cos``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [0.0, 1.0, 3.14159]
    x = jnp_zero.array(vals)
    check_allclose(jnp_zero.cos(x), jnp_zero.array([math.cos(v) for v in vals]))


def test_jnp_exp(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.exp`` against ``math.exp``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [0.0, 1.0, 2.0]
    x = jnp_zero.array(vals)
    check_allclose(jnp_zero.exp(x), jnp_zero.array([math.exp(v) for v in vals]))


def test_jnp_log(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.log`` against ``math.log``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [1.0, 2.0, 3.0]
    x = jnp_zero.array(vals)
    check_allclose(jnp_zero.log(x), jnp_zero.array([math.log(v) for v in vals]))


def test_jnp_transpose(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.transpose`` swaps matrix axes.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(jnp_zero.transpose(x), jnp_zero.array([[1.0, 3.0], [2.0, 4.0]]))


def test_jnp_reshape(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.reshape`` reshapes 1D to 2D.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([1.0, 2.0, 3.0, 4.0])
    check_allclose(jnp_zero.reshape(x, (2, 2)), jnp_zero.array([[1.0, 2.0], [3.0, 4.0]]))


def test_jnp_broadcast_to(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.broadcast_to`` replicates along leading axes.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([1.0, 2.0])
    check_allclose(jnp_zero.broadcast_to(x, (2, 2)), jnp_zero.array([[1.0, 2.0], [1.0, 2.0]]))


def test_jnp_concatenate(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.concatenate`` joins arrays along an axis.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([1.0, 2.0])
    y = jnp_zero.array([3.0, 4.0])
    check_allclose(jnp_zero.concatenate([x, y]), jnp_zero.array([1.0, 2.0, 3.0, 4.0]))


def test_jnp_where(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.where`` selects elements conditionally.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    cond = jnp_zero.array([True, False])
    x = jnp_zero.array([1.0, 2.0])
    y = jnp_zero.array([3.0, 4.0])
    check_allclose(jnp_zero.where(cond, x, y), jnp_zero.array([1.0, 4.0]))


def test_jnp_einsum(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.einsum`` computes matrix multiplication.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp_zero.array([[5.0, 6.0], [7.0, 8.0]])
    check_allclose(
        jnp_zero.einsum("ij,jk->ik", x, y),
        jnp_zero.array([[19.0, 22.0], [43.0, 50.0]]),
    )


def test_jnp_add(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.add`` computes elementwise sum.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x, y = jnp_zero.array([1.0]), jnp_zero.array([2.0])
    check_allclose(jnp_zero.add(x, y), jnp_zero.array([3.0]))


def test_jnp_multiply(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.multiply`` computes elementwise product.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x, y = jnp_zero.array([2.0]), jnp_zero.array([3.0])
    check_allclose(jnp_zero.multiply(x, y), jnp_zero.array([6.0]))


def test_jnp_maximum(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.maximum`` computes elementwise maximum.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x, y = jnp_zero.array([1.0, 5.0]), jnp_zero.array([2.0, 3.0])
    check_allclose(jnp_zero.maximum(x, y), jnp_zero.array([2.0, 5.0]))


def test_jnp_max(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.max`` reduces to maximum element.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([1.0, 2.0, 3.0])
    check_allclose(jnp_zero.max(x), jnp_zero.array(3.0))


def test_jnp_sum(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.sum`` reduces to sum of elements.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([1.0, 2.0, 3.0])
    check_allclose(jnp_zero.sum(x), jnp_zero.array(6.0))


def test_jnp_abs(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.abs`` computes elementwise absolute value.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([-1.0, 2.0])
    check_allclose(jnp_zero.abs(x), jnp_zero.array([1.0, 2.0]))


def test_jnp_mean(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.mean`` computes arithmetic mean.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([1.0, 2.0, 3.0])
    check_allclose(jnp_zero.mean(x), jnp_zero.array(2.0))


def test_jnp_dot(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.dot`` computes vector inner product.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x, y = jnp_zero.array([1.0, 2.0]), jnp_zero.array([3.0, 4.0])
    check_allclose(jnp_zero.dot(x, y), jnp_zero.array(11.0))


def test_jnp_matmul(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.matmul`` computes matrix product.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp_zero.array([[5.0, 6.0], [7.0, 8.0]])
    check_allclose(jnp_zero.matmul(x, y), jnp_zero.array([[19.0, 22.0], [43.0, 50.0]]))


def test_jnp_expand_dims(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.expand_dims`` inserts a singleton axis.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([1.0, 2.0])
    check_allclose(jnp_zero.expand_dims(x, 0), jnp_zero.array([[1.0, 2.0]]))


def test_jnp_isfinite(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.isfinite`` detects finite vs non-finite elements.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([1.0, float("inf"), float("nan")])
    check_allclose(jnp_zero.isfinite(x), jnp_zero.array([True, False, False]))


def test_jnp_subtract(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.subtract`` computes elementwise difference.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x, y = jnp_zero.array([5.0]), jnp_zero.array([2.0])
    check_allclose(jnp_zero.subtract(x, y), jnp_zero.array([3.0]))


def test_jnp_divide(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.divide`` computes elementwise division.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x, y = jnp_zero.array([5.0]), jnp_zero.array([2.0])
    check_allclose(jnp_zero.divide(x, y), jnp_zero.array([2.5]))


def test_jnp_true_divide(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.true_divide`` computes floating-point division.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x, y = jnp_zero.array([5.0]), jnp_zero.array([2.0])
    check_allclose(jnp_zero.true_divide(x, y), jnp_zero.array([2.5]))


def test_jnp_floor_divide(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.floor_divide`` computes floor division.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x, y = jnp_zero.array([5.0]), jnp_zero.array([2.0])
    check_allclose(jnp_zero.floor_divide(x, y), jnp_zero.array([2.0]))


def test_jnp_power(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.power`` computes elementwise exponentiation.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x, y = jnp_zero.array([5.0]), jnp_zero.array([2.0])
    check_allclose(jnp_zero.power(x, y), jnp_zero.array([25.0]))


def test_jnp_mod(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.mod`` computes elementwise modulo.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x, y = jnp_zero.array([5.0]), jnp_zero.array([2.0])
    check_allclose(jnp_zero.mod(x, y), jnp_zero.array([1.0]))


def test_jnp_remainder(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.remainder`` computes elementwise remainder.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x, y = jnp_zero.array([5.0]), jnp_zero.array([2.0])
    check_allclose(jnp_zero.remainder(x, y), jnp_zero.array([1.0]))


def test_jnp_divmod(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.divmod`` returns quotient and remainder tuple.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x, y = jnp_zero.array([5.0]), jnp_zero.array([2.0])
    check_allclose(jnp_zero.divmod(x, y), (jnp_zero.array([2.0]), jnp_zero.array([1.0])))


def test_jnp_negative(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.negative`` negates elements.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([5.0, -2.0])
    check_allclose(jnp_zero.negative(x), jnp_zero.array([-5.0, 2.0]))


def test_jnp_positive(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.positive`` preserves element values.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([5.0, -2.0])
    check_allclose(jnp_zero.positive(x), jnp_zero.array([5.0, -2.0]))


def test_jnp_sign(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.sign`` returns element signs.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([5.0, -2.0, 0.0])
    check_allclose(jnp_zero.sign(x), jnp_zero.array([1.0, -1.0, 0.0]))


def test_jnp_rint(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.rint`` rounds to nearest even integer.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([5.1, -2.9, 3.5])
    check_allclose(jnp_zero.rint(x), jnp_zero.array([5.0, -3.0, 4.0]))


def test_jnp_floor(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.floor`` rounds down to integer.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([5.1, -2.9, 3.5])
    check_allclose(jnp_zero.floor(x), jnp_zero.array([5.0, -3.0, 3.0]))


def test_jnp_ceil(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.ceil`` rounds up to integer.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([5.1, -2.9, 3.5])
    check_allclose(jnp_zero.ceil(x), jnp_zero.array([6.0, -2.0, 4.0]))


def test_jnp_trunc(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.trunc`` truncates toward zero.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([5.1, -2.9, 3.5])
    check_allclose(jnp_zero.trunc(x), jnp_zero.array([5.0, -2.0, 3.0]))


def test_jnp_tan(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.tan`` against ``math.tan``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [0.0, 1.0, 3.14159 / 4]
    x = jnp_zero.array(vals)
    check_allclose(jnp_zero.tan(x), jnp_zero.array([math.tan(v) for v in vals]))


def test_jnp_arcsin(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.arcsin`` against ``math.asin``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [0.0, 0.5, 1.0]
    x = jnp_zero.array(vals)
    check_allclose(jnp_zero.arcsin(x), jnp_zero.array([math.asin(v) for v in vals]))


def test_jnp_arccos(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.arccos`` against ``math.acos``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [0.0, 0.5, 1.0]
    x = jnp_zero.array(vals)
    check_allclose(jnp_zero.arccos(x), jnp_zero.array([math.acos(v) for v in vals]))


def test_jnp_arctan(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.arctan`` against ``math.atan``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [0.0, 1.0, 5.0]
    x = jnp_zero.array(vals)
    check_allclose(jnp_zero.arctan(x), jnp_zero.array([math.atan(v) for v in vals]))


def test_jnp_arctan2(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.arctan2`` against ``math.atan2``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x, y = jnp_zero.array([0.0, 1.0]), jnp_zero.array([1.0, 1.0])
    check_allclose(
        jnp_zero.arctan2(x, y),
        jnp_zero.array([math.atan2(0.0, 1.0), math.atan2(1.0, 1.0)]),
    )


def test_jnp_sinh(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.sinh`` against ``math.sinh``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [0.0, 1.0, 2.0]
    x = jnp_zero.array(vals)
    check_allclose(jnp_zero.sinh(x), jnp_zero.array([math.sinh(v) for v in vals]))


def test_jnp_cosh(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.cosh`` against ``math.cosh``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [0.0, 1.0, 2.0]
    x = jnp_zero.array(vals)
    check_allclose(jnp_zero.cosh(x), jnp_zero.array([math.cosh(v) for v in vals]))


def test_jnp_tanh(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.tanh`` against ``math.tanh``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [0.0, 1.0, 2.0]
    x = jnp_zero.array(vals)
    check_allclose(jnp_zero.tanh(x), jnp_zero.array([math.tanh(v) for v in vals]))


def test_jnp_arcsinh(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.arcsinh`` against ``math.asinh``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [0.0, 1.0, 2.0]
    x = jnp_zero.array(vals)
    check_allclose(jnp_zero.arcsinh(x), jnp_zero.array([math.asinh(v) for v in vals]))


def test_jnp_arccosh(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.arccosh`` against ``math.acosh``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [1.0, 2.0, 3.0]
    x = jnp_zero.array(vals)
    check_allclose(jnp_zero.arccosh(x), jnp_zero.array([math.acosh(v) for v in vals]))


def test_jnp_arctanh(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.arctanh`` against ``math.atanh``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [0.0, 0.5, 0.9]
    x = jnp_zero.array(vals)
    check_allclose(jnp_zero.arctanh(x), jnp_zero.array([math.atanh(v) for v in vals]))


def test_jnp_exp2(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.exp2`` against ``2 ** x``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [0.0, 1.0, 2.0]
    x = jnp_zero.array(vals)
    check_allclose(jnp_zero.exp2(x), jnp_zero.array([2.0**v for v in vals]))


def test_jnp_expm1(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.expm1`` against ``math.expm1``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [0.0, 1.0, 2.0]
    x = jnp_zero.array(vals)
    check_allclose(jnp_zero.expm1(x), jnp_zero.array([math.expm1(v) for v in vals]))


def test_jnp_log2(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.log2`` against ``math.log2``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [1.0, 2.0, 4.0]
    x = jnp_zero.array(vals)
    check_allclose(jnp_zero.log2(x), jnp_zero.array([math.log2(v) for v in vals]))


def test_jnp_log10(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.log10`` against ``math.log10``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [1.0, 10.0, 100.0]
    x = jnp_zero.array(vals)
    check_allclose(jnp_zero.log10(x), jnp_zero.array([math.log10(v) for v in vals]))


def test_jnp_log1p(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.log1p`` against ``math.log1p``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    vals = [0.0, 1.0, 2.0]
    x = jnp_zero.array(vals)
    check_allclose(jnp_zero.log1p(x), jnp_zero.array([math.log1p(v) for v in vals]))


def test_jnp_prod(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.prod`` across axes and ``keepdims``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(jnp_zero.prod(x), jnp_zero.array(24.0))
    check_allclose(jnp_zero.prod(x, axis=0), jnp_zero.array([3.0, 8.0]))
    check_allclose(jnp_zero.prod(x, axis=1, keepdims=True), jnp_zero.array([[2.0], [12.0]]))


def test_jnp_var(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.var`` across axes and ``keepdims``.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(jnp_zero.var(x), jnp_zero.array(1.25))
    check_allclose(jnp_zero.var(x, axis=0), jnp_zero.array([1.0, 1.0]))
    check_allclose(jnp_zero.var(x, axis=1, keepdims=True), jnp_zero.array([[0.25], [0.25]]))


def test_jnp_std(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.std`` across axes.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(jnp_zero.std(x), jnp_zero.array(math.sqrt(1.25)))
    check_allclose(jnp_zero.std(x, axis=0), jnp_zero.array([1.0, 1.0]))


def test_jnp_min(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.min`` across axes.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(jnp_zero.min(x), jnp_zero.array(1.0))
    check_allclose(jnp_zero.min(x, axis=0), jnp_zero.array([1.0, 2.0]))


def test_jnp_amin(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.amin`` reduces to minimum value.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(jnp_zero.amin(x), jnp_zero.array(1.0))


def test_jnp_amax(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.amax`` reduces to maximum value.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(jnp_zero.amax(x), jnp_zero.array(4.0))


def test_jnp_argmax(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.argmax`` finds maximum indices.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(jnp_zero.argmax(x), jnp_zero.array(3))
    check_allclose(jnp_zero.argmax(x, axis=1), jnp_zero.array([1, 1]))


def test_jnp_argmin(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.argmin`` finds minimum indices.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(jnp_zero.argmin(x), jnp_zero.array(0))
    check_allclose(jnp_zero.argmin(x, axis=0), jnp_zero.array([0, 0]))


def test_jnp_any(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.any`` computes logical OR reduction.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[True, False], [False, False]])
    check_allclose(jnp_zero.any(x), jnp_zero.array(True))
    check_allclose(jnp_zero.any(x, axis=0), jnp_zero.array([True, False]))


def test_jnp_all(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.all`` computes logical AND reduction.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[True, False], [True, True]])
    check_allclose(jnp_zero.all(x), jnp_zero.array(False))
    check_allclose(jnp_zero.all(x, axis=0), jnp_zero.array([True, False]))


def test_jnp_ravel(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.ravel`` flattens an array to 1D.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[1.0, 2.0], [3.0, 4.0]])
    check_allclose(jnp_zero.ravel(x), jnp_zero.array([1.0, 2.0, 3.0, 4.0]))


def test_jnp_squeeze(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.squeeze`` removes singleton dimensions.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[[1.0, 2.0]]])
    check_allclose(jnp_zero.squeeze(x), jnp_zero.array([1.0, 2.0]))


def test_jnp_swapaxes(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.swapaxes`` exchanges two axes.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    check_allclose(
        jnp_zero.swapaxes(x, 0, 2),
        jnp_zero.array([[[1, 5], [3, 7]], [[2, 6], [4, 8]]]),
    )


def test_jnp_moveaxis(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.moveaxis`` moves an axis to a new position.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    check_allclose(
        jnp_zero.moveaxis(x, 0, -1),
        jnp_zero.array([[[1, 5], [2, 6]], [[3, 7], [4, 8]]]),
    )


def test_jnp_stack(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.stack`` stacks arrays along a new axis.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    a = jnp_zero.array([1, 2, 3])
    b = jnp_zero.array([4, 5, 6])
    check_allclose(jnp_zero.stack((a, b)), jnp_zero.array([[1, 2, 3], [4, 5, 6]]))
    check_allclose(
        jnp_zero.stack((a, b), axis=1),
        jnp_zero.array([[1, 4], [2, 5], [3, 6]]),
    )


def test_jnp_vstack(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.vstack`` stacks arrays vertically.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    a = jnp_zero.array([1, 2, 3])
    b = jnp_zero.array([4, 5, 6])
    check_allclose(jnp_zero.vstack((a, b)), jnp_zero.array([[1, 2, 3], [4, 5, 6]]))


def test_jnp_hstack(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.hstack`` stacks arrays horizontally.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    a = jnp_zero.array([1, 2, 3])
    b = jnp_zero.array([4, 5, 6])
    check_allclose(jnp_zero.hstack((a, b)), jnp_zero.array([1, 2, 3, 4, 5, 6]))


def test_jnp_dstack(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.dstack`` stacks arrays along depth axis.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    a = jnp_zero.array([1, 2, 3])
    b = jnp_zero.array([4, 5, 6])
    check_allclose(
        jnp_zero.dstack((a, b)),
        jnp_zero.array([[[1, 4], [2, 5], [3, 6]]]),
    )


def test_jnp_split(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.split`` splits an array into equal sub-arrays.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.arange(9.0)
    check_allclose(
        jnp_zero.split(x, 3),
        [
            jnp_zero.array([0.0, 1.0, 2.0]),
            jnp_zero.array([3.0, 4.0, 5.0]),
            jnp_zero.array([6.0, 7.0, 8.0]),
        ],
    )


def test_jnp_array_split(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.array_split`` splits an array into uneven sub-arrays.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.arange(8.0)
    check_allclose(
        jnp_zero.array_split(x, 3),
        [
            jnp_zero.array([0.0, 1.0, 2.0]),
            jnp_zero.array([3.0, 4.0, 5.0]),
            jnp_zero.array([6.0, 7.0]),
        ],
    )


def test_jnp_vsplit(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.vsplit`` splits a 2D array vertically.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.reshape(jnp_zero.arange(16.0), (4, 4))
    check_allclose(
        jnp_zero.vsplit(x, 2),
        [
            jnp_zero.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]),
            jnp_zero.array([[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]]),
        ],
    )


def test_jnp_hsplit(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.hsplit`` splits a 2D array horizontally.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.reshape(jnp_zero.arange(16.0), (4, 4))
    check_allclose(
        jnp_zero.hsplit(x, 2),
        [
            jnp_zero.array([[0.0, 1.0], [4.0, 5.0], [8.0, 9.0], [12.0, 13.0]]),
            jnp_zero.array([[2.0, 3.0], [6.0, 7.0], [10.0, 11.0], [14.0, 15.0]]),
        ],
    )


def test_jnp_dsplit(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.dsplit`` splits a 3D array along depth.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    x = jnp_zero.reshape(jnp_zero.arange(16.0), (2, 2, 4))
    check_allclose(
        jnp_zero.dsplit(x, 2),
        [
            jnp_zero.array([[[0.0, 1.0], [4.0, 5.0]], [[8.0, 9.0], [12.0, 13.0]]]),
            jnp_zero.array([[[2.0, 3.0], [6.0, 7.0]], [[10.0, 11.0], [14.0, 15.0]]]),
        ],
    )


def test_jnp_tile(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.tile`` repeats an array according to reps.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    a = jnp_zero.array([0, 1, 2])
    check_allclose(jnp_zero.tile(a, 2), jnp_zero.array([0, 1, 2, 0, 1, 2]))


def test_jnp_repeat(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.repeat`` repeats each element of an array.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    a = jnp_zero.array([3, 4])
    check_allclose(jnp_zero.repeat(a, 2), jnp_zero.array([3, 3, 4, 4]))


def test_jnp_pad(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.pad`` pads an array with zeros.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    a = jnp_zero.array([1.0, 2.0, 3.0])
    check_allclose(jnp_zero.pad(a, (1, 2)), jnp_zero.array([0.0, 1.0, 2.0, 3.0, 0.0, 0.0]))


def test_jnp_take(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.take`` gathers elements by flat index.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    a = jnp_zero.array([4, 3, 5, 7, 6, 8])
    indices = jnp_zero.array([0, 1, 4])
    check_allclose(jnp_zero.take(a, indices), jnp_zero.array([4, 3, 6]))


def test_jnp_take_along_axis(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.take_along_axis`` gathers along a specified axis.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    a = jnp_zero.array([[10, 30, 20], [60, 40, 50]])
    indices = jnp_zero.array([[1, 0, 2], [2, 0, 1]])
    check_allclose(
        jnp_zero.take_along_axis(a, indices, axis=1),
        jnp_zero.array([[30, 10, 20], [50, 60, 40]]),
    )


def test_jnp_vdot(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.vdot`` computes vector dot product.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    a = jnp_zero.array([1, 2, 3])
    b = jnp_zero.array([0, 1, 0])
    check_allclose(jnp_zero.vdot(a, b), jnp_zero.array(2))


def test_jnp_inner(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.inner`` computes inner product.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    a = jnp_zero.array([1, 2, 3])
    b = jnp_zero.array([0, 1, 0])
    check_allclose(jnp_zero.inner(a, b), jnp_zero.array(2))


def test_jnp_outer(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.outer`` computes outer product.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    a = jnp_zero.array([1, 2, 3])
    b = jnp_zero.array([0, 1, 0])
    check_allclose(
        jnp_zero.outer(a, b),
        jnp_zero.array([[0, 1, 0], [0, 2, 0], [0, 3, 0]]),
    )


def test_jnp_tensordot(check_allclose: VariadicCallable[None]) -> None:
    """Verify ``zero_jax.numpy.tensordot`` contracts specified axes.

    Args:
        check_allclose: Numerical closeness assertion fixture.
    """
    a = jnp_zero.reshape(jnp_zero.arange(60.0), (3, 4, 5))
    b = jnp_zero.reshape(jnp_zero.arange(24.0), (4, 3, 2))
    expected = jnp_zero.array(
        [
            [4400.0, 4730.0],
            [4532.0, 4874.0],
            [4664.0, 5018.0],
            [4796.0, 5162.0],
            [4928.0, 5306.0],
        ]
    )
    check_allclose(
        jnp_zero.tensordot(a, b, axes=([1, 0], [0, 1])),
        expected,
    )
