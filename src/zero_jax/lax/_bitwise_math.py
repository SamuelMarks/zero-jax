"""LAX bitwise, rounding, complex, precision, and special mathematical primitives."""

from __future__ import annotations

from ml_switcheroo_compiler import random as _compiler_random
from ml_switcheroo_compiler.core.dtype import DType

import zero_jax._compiler_proxy_ops as ops
from zero_jax._types import ArrayLikeSpec
from zero_jax.lax.primitives import cbrt, digamma, erf, erfc, lgamma, rsqrt, shift_left, shift_right_arithmetic, shift_right_logical
from zero_jax.numpy import (
    array,
    asarray,
    astype,
    bitwise_right_shift,
    exp,
    int64,
    isfinite,
    polyval,
    power,
    square,
)
from zero_jax.numpy import (
    log2 as jnp_log2,
)
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap, ndarray


def regularized_incomplete_beta(a: ArrayLikeSpec, b: ArrayLikeSpec, x: ArrayLikeSpec) -> ndarray:
    """Compute the regularized incomplete beta function ``I_x(a, b)``.

    Args:
        a: First shape parameter.
        b: Second shape parameter.
        x: Upper limit of integration in ``[0, 1]``.

    Returns:
        ndarray: Regularized incomplete beta values.
    """
    return betainc(a, b, x)


def bessel_i0e(x: ArrayLikeSpec) -> ndarray:
    """Compute the exponentially scaled modified Bessel function of order 0.

    Args:
        x: Input parameter ``x``.

    Returns:
        ndarray: Evaluated ``i0e(x)``.
    """
    return _wrap(ops.bessel_i0e(_to_tensor(x)))


def bessel_i1e(x: ArrayLikeSpec) -> ndarray:
    """Compute the exponentially scaled modified Bessel function of order 1.

    Args:
        x: Input parameter ``x``.

    Returns:
        ndarray: Evaluated ``i1e(x)``.
    """
    return _wrap(ops.bessel_i1e(_to_tensor(x)))


def betainc(a: ArrayLikeSpec, b: ArrayLikeSpec, x: ArrayLikeSpec) -> ndarray:
    """Compute the regularized incomplete beta integral ``I_x(a, b)``.

    Args:
        a: Input parameter ``a``.
        b: Input parameter ``b``.
        x: Input parameter ``x``.

    Returns:
        ndarray: Regularized incomplete beta values.
    """
    return _wrap(ops.betainc(_to_tensor(a), _to_tensor(b), _to_tensor(x)))


def bitcast_convert_type(operand: ArrayLikeSpec, new_dtype: ArrayLikeSpec) -> ndarray:
    """Bitcast ``operand`` to ``new_dtype``.

    Args:
        operand: Input parameter ``operand``.
        new_dtype: Target dtype specification.

    Returns:
        ndarray: Bitcast array.
    """
    return _wrap(ops.bitcast_convert_type(_to_tensor(operand), new_dtype))


def clz(x: ArrayLikeSpec) -> ndarray:
    """Count leading zeros in integer elements of ``x``.

    Args:
        x: Input integer array.

    Returns:
        ndarray: Leading zero counts.
    """
    return _wrap(ops.clz(_to_tensor(x)))


def complex(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ndarray:
    """Construct a complex array ``x + 1j * y`` from real and imaginary parts.

    Args:
        x: Real component array.
        y: Imaginary component array.

    Returns:
        ndarray: Complex array.
    """
    return array(x) + 1.0j * array(y)


def convert_element_type(operand: ArrayLikeSpec, new_dtype: ArrayLikeSpec) -> ndarray:
    """Convert elements of ``operand`` to ``new_dtype``.

    Args:
        operand: Input array.
        new_dtype: Target dtype.

    Returns:
        ndarray: Converted array.
    """
    return astype(operand, new_dtype)


def erf_inv(x: ArrayLikeSpec) -> ndarray:
    """Compute the inverse error function elementwise.

    Args:
        x: Input array in ``(-1, 1)``.

    Returns:
        ndarray: Inverse error function values.
    """
    return _wrap(ops.erf_inv(_to_tensor(x)))


def igamma(a: ArrayLikeSpec, x: ArrayLikeSpec) -> ndarray:
    """Compute the regularized lower incomplete gamma function ``P(a, x)``.

    Args:
        a: Shape parameter ``a``.
        x: Integration limit ``x``.

    Returns:
        ndarray: Regularized lower incomplete gamma values.
    """
    return _wrap(ops.igamma(_to_tensor(a), _to_tensor(x)))


def igamma_grad_a(a: ArrayLikeSpec, x: ArrayLikeSpec) -> ndarray:
    """Compute the derivative of ``igamma(a, x)`` with respect to ``a``.

    Args:
        a: Shape parameter ``a``.
        x: Integration limit ``x``.

    Returns:
        ndarray: Partial derivative ``d/da igamma(a, x)``.
    """
    return _wrap(ops.igamma_grad_a(_to_tensor(a), _to_tensor(x)))


def igammac(a: ArrayLikeSpec, x: ArrayLikeSpec) -> ndarray:
    """Compute the regularized upper incomplete gamma function ``Q(a, x)``.

    Args:
        a: Shape parameter ``a``.
        x: Integration limit ``x``.

    Returns:
        ndarray: Regularized upper incomplete gamma values.
    """
    return _wrap(ops.igammac(_to_tensor(a), _to_tensor(x)))


def integer_pow(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ndarray:
    """Raise ``x`` to integer power ``y``.

    Args:
        x: Base array.
        y: Integer exponent.

    Returns:
        ndarray: ``x ** y``.
    """
    return power(x, y)


def is_finite(x: ArrayLikeSpec) -> ndarray:
    """Test elementwise whether ``x`` is finite.

    Args:
        x: Input array.

    Returns:
        ndarray: Boolean array indicating finite entries.
    """
    return isfinite(x)


def log2(x: ArrayLikeSpec, accuracy: ArrayLikeSpec = None) -> ndarray:
    """Compute elementwise base-2 logarithm of ``x``.

    Args:
        x: Input array.
        accuracy: Optional accuracy mode specification.

    Returns:
        ndarray: Base-2 logarithm of ``x``.
    """
    _ = accuracy
    return jnp_log2(x)


def logistic(x: ArrayLikeSpec) -> ndarray:
    """Compute the standard logistic sigmoid ``1 / (1 + exp(-x))``.

    Args:
        x: Input array.

    Returns:
        ndarray: Logistic sigmoid of ``x``.
    """
    return 1.0 / (1.0 + exp(-x))


def mulhi(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ndarray:
    """Compute the high half of the full-width integer product of ``x`` and ``y``.

    Args:
        x: First integer operand.
        y: Second integer operand.

    Returns:
        ndarray: High bits of ``x * y``.
    """
    prod_64 = asarray(x, dtype=int64) * asarray(y, dtype=int64)
    return bitwise_right_shift(prod_64, 32)


def one_minus_square(x: ArrayLikeSpec) -> ndarray:
    """Compute ``1 - x ** 2`` elementwise.

    Args:
        x: Input array.

    Returns:
        ndarray: ``1 - x ** 2``.
    """
    arr = asarray(x)
    return 1 - square(arr)


def polygamma(n: ArrayLikeSpec, x: ArrayLikeSpec) -> ndarray:
    """Compute the ``n``-th derivative of the digamma function at ``x``.

    Args:
        n: Derivative order ``n``.
        x: Evaluation point ``x``.

    Returns:
        ndarray: Evaluated polygamma ``psi^(n)(x)``.
    """
    return _wrap(ops.polygamma(_to_tensor(n), _to_tensor(x)))


def polynomial(x: ArrayLikeSpec, coeffs: ArrayLikeSpec, *, unroll: int | None = None) -> ndarray:
    """Evaluate a polynomial with coefficients ``coeffs`` at ``x`` using Horner's method.

    Args:
        x: Evaluation point array.
        coeffs: Sequence of polynomial coefficients from highest to lowest degree.
        unroll: Optional loop unroll threshold.

    Returns:
        ndarray: Evaluated polynomial values.
    """
    _ = unroll
    return polyval(coeffs, x)


def population_count(operand: ArrayLikeSpec) -> ndarray:
    """Count the number of set bits (popcount) in each element of ``operand``.

    Args:
        operand: Input integer array.

    Returns:
        ndarray: Elementwise bit count.
    """
    return _wrap(ops.population_count(_to_tensor(operand)))


def pow(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ndarray:
    """Compute ``x ** y`` elementwise.

    Args:
        x: Base array.
        y: Exponent array.

    Returns:
        ndarray: ``x ** y``.
    """
    return power(x, y)


def random_gamma_grad(a: ArrayLikeSpec, x: ArrayLikeSpec) -> ndarray:
    """Compute the implicit reparameterization gradient for Gamma random variables.

    Args:
        a: Gamma shape parameter ``a``.
        x: Gamma sample ``x``.

    Returns:
        ndarray: Reparameterization gradient ``dx/da``.
    """
    return _wrap(ops.random_gamma_grad(_to_tensor(a), _to_tensor(x)))


def rem(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ndarray:
    """Compute the remainder of division ``x % y`` with sign matching dividend ``x``.

    Args:
        x: Dividend array.
        y: Divisor array.

    Returns:
        ndarray: Remainder array.
    """
    return _wrap(ops.rem(_to_tensor(x), _to_tensor(y)))


def rng_bit_generator(key: ArrayLikeSpec, shape: ArrayLikeSpec) -> tuple[ndarray, ndarray]:
    """Generate uniform random bits of ``shape`` and an updated PRNG ``key``.

    Args:
        key: Input PRNG key array.
        shape: Output shape tuple or integer sequence.

    Returns:
        tuple[ndarray, ndarray]: Pair of ``(new_key, random_bits)``.
    """
    target_shape = (max(1, int(shape)),) if isinstance(shape, (int, float)) else tuple(max(1, int(s)) for s in asarray(shape).__array__().ravel())
    key_t = _to_tensor(key)
    new_key = _wrap(ops.add(key_t, _to_tensor(1)))
    bits_out = _wrap(_compiler_random.randint(key_t, shape=target_shape, minval=0, maxval=65536, dtype=DType.Int32))
    return new_key, bits_out


def rng_uniform(a: ArrayLikeSpec, b: ArrayLikeSpec, shape: ArrayLikeSpec) -> ndarray:
    """Sample uniform random floating-point values in ``[a, b)`` of ``shape``.

    Args:
        a: Lower bound scalar or array.
        b: Upper bound scalar or array.
        shape: Output shape tuple or integer sequence.

    Returns:
        ndarray: Uniformly distributed samples in ``[a, b)``.
    """
    target_shape = (max(1, int(shape)),) if isinstance(shape, (int, float)) else tuple(max(1, int(s)) for s in asarray(shape).__array__().ravel())
    a_t = _to_tensor(a)
    b_t = _to_tensor(b)
    u_t = _compiler_random.uniform(a_t, shape=target_shape, dtype=a_t.dtype, minval=0.0, maxval=1.0)
    return _wrap(ops.add(a_t, ops.multiply(ops.subtract(b_t, a_t), u_t)))


def zeta(x: ArrayLikeSpec, q: ArrayLikeSpec) -> ndarray:
    """Compute the Hurwitz zeta function ``zeta(x, q)``.

    Args:
        x: Exponent parameter ``x``.
        q: Offset parameter ``q``.

    Returns:
        ndarray: Evaluated Hurwitz zeta values.
    """
    return _wrap(ops.zeta(_to_tensor(x), _to_tensor(q)))


__all__ = [
    "bessel_i0e",
    "bessel_i1e",
    "betainc",
    "bitcast_convert_type",
    "cbrt",
    "clz",
    "complex",
    "convert_element_type",
    "digamma",
    "erf",
    "erf_inv",
    "erfc",
    "igamma",
    "igamma_grad_a",
    "igammac",
    "integer_pow",
    "is_finite",
    "lgamma",
    "log2",
    "logistic",
    "mulhi",
    "one_minus_square",
    "polygamma",
    "polynomial",
    "population_count",
    "pow",
    "random_gamma_grad",
    "regularized_incomplete_beta",
    "rem",
    "rng_bit_generator",
    "rng_uniform",
    "rsqrt",
    "shift_left",
    "shift_right_arithmetic",
    "shift_right_logical",
    "zeta",
]
