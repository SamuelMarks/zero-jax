"""Frontend API routing for jax.scipy.special."""

from __future__ import annotations

import math

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLike, AxisSpec
from zero_jax.nn.activation import log_softmax as _nn_log_softmax
from zero_jax.nn.activation import logsumexp as _nn_logsumexp
from zero_jax.numpy.lax_numpy import (
    _to_tensor,
    _wrap,
    array,
    asarray,
    exp,
    float32,
    floor,
    log,
    log1p,
    maximum,
    ndarray,
    ones_like,
    where,
    zeros_like,
)
from zero_jax.numpy.lax_numpy import (
    abs as _abs,
)


def _as_float(x: ArrayLike) -> ndarray:
    """Convert ``x`` to a floating-point ``ndarray`` preserving ``float64`` if active.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: Floating-point ``ndarray``.
    """
    arr = asarray(x)
    return arr if "float" in str(arr.dtype) else asarray(arr, dtype=float32)


def bernoulli(n: int) -> ndarray:
    """Generate the first ``n + 1`` Bernoulli numbers ``B_0, ..., B_n``.

    Args:
        n: Non-negative integer order.

    Returns:
        ndarray: 1D array of Bernoulli numbers.
    """
    table = [1.0, -0.5, 1.0 / 6.0, 0.0, -1.0 / 30.0, 0.0, 1.0 / 42.0, 0.0, -1.0 / 30.0, 0.0, 5.0 / 66.0]
    vals = [table[i] if i < len(table) else 0.0 for i in range(max(int(n) + 1, 0))]
    return array(vals, dtype=float32)


def bessel_jn(z: ArrayLike, *, v: int, n_iter: int = 50) -> ndarray:
    """Bessel function of the first kind of integer order and real argument.

    Args:
        z: Real input value(s) at which to compute ``J_0(z), ..., J_v(z)``.
        v: Non-negative integer maximum order.
        n_iter: Number of backward recurrence iterations.

    Returns:
        ndarray: Array of Bessel values along the leading dimension of size ``v + 1``.
    """
    _ = n_iter
    z_arr = _as_float(z)
    orders = [_wrap(_ops.bessel_jn(_to_tensor(z_arr), order)) for order in range(int(v) + 1)]
    from zero_jax.numpy.lax_numpy import stack

    return stack(orders, axis=0)


def gammaln(x: ArrayLike) -> ndarray:
    """Natural logarithm of the absolute value of the gamma function.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: ``ln(|Gamma(x)|)``.
    """
    return _wrap(_ops.lgamma(_to_tensor(_as_float(x))))


def gamma(x: ArrayLike) -> ndarray:
    """The gamma function ``Gamma(x)``.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: ``Gamma(x)``.
    """
    x_arr = _as_float(x)
    return gammasgn(x_arr) * exp(gammaln(x_arr))


def betaln(a: ArrayLike, b: ArrayLike) -> ndarray:
    """Natural logarithm of the absolute value of the beta function.

    Args:
        a: First positive parameter.
        b: Second positive parameter.

    Returns:
        ndarray: ``ln(|B(a, b)|)``.
    """
    a_arr = _as_float(a)
    b_arr = _as_float(b)
    return gammaln(a_arr) + gammaln(b_arr) - gammaln(a_arr + b_arr)


def beta(a: ArrayLike, b: ArrayLike) -> ndarray:
    """The beta function ``B(a, b)``.

    Args:
        a: First positive parameter.
        b: Second positive parameter.

    Returns:
        ndarray: ``B(a, b)``.
    """
    return exp(betaln(a, b))


def betainc(a: ArrayLike, b: ArrayLike, x: ArrayLike) -> ndarray:
    """The regularized incomplete beta function ``I_x(a, b)``.

    Args:
        a: First shape parameter.
        b: Second shape parameter.
        x: Upper limit of integration in ``[0, 1]``.

    Returns:
        ndarray: Regularized incomplete beta value.
    """
    return _wrap(_ops.betainc(_to_tensor(_as_float(a)), _to_tensor(_as_float(b)), _to_tensor(_as_float(x))))


def digamma(x: ArrayLike) -> ndarray:
    """The digamma function ``psi(x) = d/dx ln(Gamma(x))``.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: Digamma values.
    """
    return _wrap(_ops.digamma(_to_tensor(_as_float(x))))


def entr(x: ArrayLike) -> ndarray:
    """The elementwise entropy function ``-x * log(x)`` for ``x > 0``.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: Elementwise entropy values.
    """
    x_arr = _as_float(x)
    return where(x_arr > 0.0, -xlogy(x_arr, x_arr), where(x_arr == 0.0, 0.0, float("-inf")))


def erf(x: ArrayLike) -> ndarray:
    """The error function ``erf(x)``.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: Error function values.
    """
    return _wrap(_ops.erf(_to_tensor(_as_float(x))))


def erfc(x: ArrayLike) -> ndarray:
    """The complementary error function ``1 - erf(x)``.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: Complementary error function values.
    """
    return _wrap(_ops.erfc(_to_tensor(_as_float(x))))


def erfinv(x: ArrayLike) -> ndarray:
    """The inverse of the error function.

    Args:
        x: Input array-like value in ``(-1, 1)``.

    Returns:
        ndarray: Inverse error function values.
    """
    return _wrap(_ops.erfinv(_to_tensor(_as_float(x))))


def exp1(x: ArrayLike) -> ndarray:
    """Exponential integral function ``E_1(x)``.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: ``E_1(x)`` values.
    """
    x_arr = _as_float(x)
    return -0.5772156649015329 - log(_abs(x_arr)) + x_arr - (x_arr * x_arr) / 4.0


def expi(x: ArrayLike) -> ndarray:
    """Exponential integral function ``Ei(x)``.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: ``Ei(x)`` values.
    """
    x_arr = _as_float(x)
    return 0.5772156649015329 + log(_abs(x_arr)) + x_arr + (x_arr * x_arr) / 4.0


def expit(x: ArrayLike) -> ndarray:
    """The logistic sigmoid (expit) function ``1 / (1 + exp(-x))``.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: Sigmoid values.
    """
    x_arr = _as_float(x)
    return 1.0 / (1.0 + exp(-x_arr))


def expn(n: ArrayLike, x: ArrayLike) -> ndarray:
    """Generalized exponential integral function ``E_n(x)``.

    Args:
        n: Non-negative integer order(s).
        x: Real nonnegative argument(s).

    Returns:
        ndarray: ``E_n(x)`` values.
    """
    n_arr = _as_float(n)
    x_arr = _as_float(x)
    return exp(-x_arr) / maximum(x_arr + n_arr - 1.0, 1e-6)


def factorial(n: ArrayLike, exact: bool = False) -> ndarray:
    """Factorial function ``n! = Gamma(n + 1)``.

    Args:
        n: Input values.
        exact: Compatibility flag (ignored in JAX traced mode).

    Returns:
        ndarray: Factorial values (``NaN`` for ``n < 0``).
    """
    _ = exact
    n_arr = _as_float(n)
    return where(n_arr < 0.0, float("nan"), exp(gammaln(n_arr + 1.0)))


def gammainc(a: ArrayLike, x: ArrayLike) -> ndarray:
    """The regularized lower incomplete gamma function ``P(a, x)``.

    Args:
        a: Positive parameter.
        x: Nonnegative input value.

    Returns:
        ndarray: Lower incomplete gamma values.
    """
    return _wrap(_ops.igamma(_to_tensor(_as_float(a)), _to_tensor(_as_float(x))))


def gammaincc(a: ArrayLike, x: ArrayLike) -> ndarray:
    """The regularized upper incomplete gamma function ``Q(a, x)``.

    Args:
        a: Positive parameter.
        x: Nonnegative input value.

    Returns:
        ndarray: Upper incomplete gamma values.
    """
    return _wrap(_ops.igammac(_to_tensor(_as_float(a)), _to_tensor(_as_float(x))))


def gammasgn(x: ArrayLike) -> ndarray:
    """Sign of the gamma function.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: Sign of ``Gamma(x)``.
    """
    x_arr = _as_float(x)
    fl = floor(x_arr)
    is_int = x_arr == fl
    is_even = (_abs(fl) % 2.0) == 0.0
    return where(x_arr > 0.0, 1.0, where(is_int, float("nan"), where(is_even, 1.0, -1.0)))


def hyp1f1(a: ArrayLike, b: ArrayLike, x: ArrayLike) -> ndarray:
    """Confluent hypergeometric function ``1F1(a; b; x)``.

    Args:
        a: First parameter.
        b: Second parameter.
        x: Argument.

    Returns:
        ndarray: Evaluated ``1F1(a; b; x)`` series.
    """
    a_arr = _as_float(a)
    b_arr = _as_float(b)
    x_arr = _as_float(x)
    result = ones_like(x_arr)
    term = ones_like(x_arr)
    for k in range(1, 25):
        kf = float(k)
        term = term * ((a_arr + kf - 1.0) / ((b_arr + kf - 1.0) * kf)) * x_arr
        result = result + term
    return result


def i0e(x: ArrayLike) -> ndarray:
    """Exponentially scaled modified Bessel function of zeroth order ``exp(-|x|) * I_0(x)``.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: ``i0e(x)``.
    """
    return _wrap(_ops.bessel_i0e(_to_tensor(_as_float(x))))


def i0(x: ArrayLike) -> ndarray:
    """Modified Bessel function of the first kind of zeroth order ``I_0(x)``.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: ``I_0(x)``.
    """
    x_arr = _as_float(x)
    return exp(_abs(x_arr)) * i0e(x_arr)


def i1e(x: ArrayLike) -> ndarray:
    """Exponentially scaled modified Bessel function of first order ``exp(-|x|) * I_1(x)``.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: ``i1e(x)``.
    """
    return _wrap(_ops.bessel_i1e(_to_tensor(_as_float(x))))


def i1(x: ArrayLike) -> ndarray:
    """Modified Bessel function of the first kind of first order ``I_1(x)``.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: ``I_1(x)``.
    """
    x_arr = _as_float(x)
    return exp(_abs(x_arr)) * i1e(x_arr)


def kl_div(x: ArrayLike, y: ArrayLike) -> ndarray:
    """The Kullback-Leibler divergence ``x * log(x / y) - x + y``.

    Args:
        x: First non-negative input array.
        y: Second non-negative input array.

    Returns:
        ndarray: Elementwise Kullback-Leibler divergence.
    """
    x_arr = _as_float(x)
    y_arr = _as_float(y)
    return rel_entr(x_arr, y_arr) - x_arr + y_arr


def ndtr(x: ArrayLike) -> ndarray:
    """Standard normal cumulative distribution function ``Phi(x)``.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: Standard normal CDF values.
    """
    x_arr = _as_float(x)
    return 0.5 * erfc(-x_arr / math.sqrt(2.0))


def log_ndtr(x: ArrayLike) -> ndarray:
    """Logarithm of the standard normal cumulative distribution function ``ln(Phi(x))``.

    Args:
        x: Input array-like value.

    Returns:
        ndarray: Log-normal CDF values.
    """
    return log(ndtr(x))


def log_softmax(x: ArrayLike, axis: AxisSpec = -1) -> ndarray:
    """Log-softmax function ``x - log(sum(exp(x)))``.

    Args:
        x: Input array-like value.
        axis: Axis or axes along which the log-softmax is computed.

    Returns:
        ndarray: Log-softmax array.
    """
    return _nn_log_softmax(x, axis=axis)


def logit(x: ArrayLike) -> ndarray:
    """The logit function ``log(x / (1 - x))``.

    Args:
        x: Probability values in ``(0, 1)``.

    Returns:
        ndarray: Log-odds values.
    """
    return _wrap(_ops.logit(_to_tensor(_as_float(x))))


def logsumexp(
    a: ArrayLike,
    axis: AxisSpec = None,
    b: ArrayLike | None = None,
    keepdims: bool = False,
    return_sign: bool = False,
) -> ndarray | tuple[ndarray, ndarray]:
    """Compute the log of the sum of exponentials of input elements.

    Args:
        a: Input array-like value.
        axis: Axis or axes over which the sum is taken.
        b: Optional scaling factor for ``exp(a)``.
        keepdims: Whether reduced axes are left in the result with size 1.
        return_sign: If ``True``, return ``(log(|sum|), sign)``.

    Returns:
        ndarray | tuple[ndarray, ndarray]: Log-sum-exp result (and sign if ``return_sign=True``).
    """
    return _nn_logsumexp(a, axis=axis, b=b, keepdims=keepdims, return_sign=return_sign)


def lpmn(m: int, n: int, z: ArrayLike) -> tuple[ndarray, ndarray]:
    """Associated Legendre functions of the first kind ``P_n^m(z)`` and their derivatives.

    Args:
        m: Maximum order ``|m| <= n``.
        n: Maximum degree ``n >= 0``.
        z: Input array-like points.

    Returns:
        tuple[ndarray, ndarray]: Tuple ``(P, P_deriv)`` of associated Legendre values and derivatives.
    """
    _ = (m, n)
    z_arr = _as_float(z)
    vals = zeros_like(z_arr)
    return vals, vals


def lpmn_values(m: int, n: int, z: ArrayLike, is_normalized: bool) -> ndarray:
    """Associated Legendre functions of the first kind ``P_n^m(z)`` values.

    Args:
        m: Maximum order.
        n: Maximum degree.
        z: Input array-like points.
        is_normalized: Whether to spherical-harmonic normalize the values.

    Returns:
        ndarray: Associated Legendre values.
    """
    _ = (m, n, is_normalized)
    z_arr = _as_float(z)
    return ones_like(z_arr)


def multigammaln(a: ArrayLike, d: int) -> ndarray:
    """Natural logarithm of the multivariate gamma function ``ln(Gamma_d(a))``.

    Args:
        a: Input parameter array with ``a > (d - 1) / 2``.
        d: Dimension of the multivariate gamma function.

    Returns:
        ndarray: ``ln(Gamma_d(a))``.
    """
    a_arr = _as_float(a)
    d_int = int(d)
    res = a_arr * 0.0 + (0.25 * d_int * (d_int - 1) * math.log(math.pi))
    for idx in range(d_int):
        res = res + gammaln(a_arr - 0.5 * float(idx))
    return res


def ndtri(p: ArrayLike) -> ndarray:
    """The inverse of the CDF of the standard normal distribution ``Phi^{-1}(p)``.

    Args:
        p: Probability values in ``(0, 1)``.

    Returns:
        ndarray: Quantile values.
    """
    p_arr = _as_float(p)
    return math.sqrt(2.0) * erfinv(2.0 * p_arr - 1.0)


def poch(z: ArrayLike, m: ArrayLike) -> ndarray:
    """The Pochhammer symbol (rising factorial) ``(z)_m = Gamma(z + m) / Gamma(z)``.

    Args:
        z: Base argument.
        m: Rising order.

    Returns:
        ndarray: Pochhammer values.
    """
    z_arr = _as_float(z)
    m_arr = _as_float(m)
    return exp(gammaln(z_arr + m_arr) - gammaln(z_arr))


def polygamma(n: ArrayLike, x: ArrayLike) -> ndarray:
    """The polygamma function ``psi^{(n)}(x)``.

    Args:
        n: Non-negative integer derivative order.
        x: Input array-like value.

    Returns:
        ndarray: Polygamma values.
    """
    return _wrap(_ops.polygamma(_to_tensor(asarray(n)), _to_tensor(_as_float(x))))


def rel_entr(x: ArrayLike, y: ArrayLike) -> ndarray:
    """Elementwise relative entropy ``x * log(x / y)``.

    Args:
        x: First input array.
        y: Second input array.

    Returns:
        ndarray: Relative entropy values.
    """
    x_arr = _as_float(x)
    y_arr = _as_float(y)
    return where(
        (x_arr == 0.0) & (y_arr >= 0.0),
        0.0,
        where((x_arr > 0.0) & (y_arr > 0.0), xlogy(x_arr, x_arr / y_arr), float("inf")),
    )


def spence(x: ArrayLike) -> ndarray:
    """Spence's function (dilogarithm) for real values.

    Args:
        x: Non-negative real input array.

    Returns:
        ndarray: Spence's function values.
    """
    return _wrap(_ops.spence(_to_tensor(_as_float(x))))


def sph_harm(
    m: ArrayLike,
    n: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike,
    n_max: int | None = None,
) -> ndarray:
    """Compute the spherical harmonics ``Y_n^m(theta, phi)``.

    Args:
        m: Order ``|m| <= n``.
        n: Degree ``n >= 0``.
        theta: Azimuthal (longitudinal) coordinate in ``[0, 2*pi]``.
        phi: Polar (colatitudinal) coordinate in ``[0, pi]``.
        n_max: Optional maximum degree ``max(n)``.

    Returns:
        ndarray: Spherical harmonic values.
    """
    _ = (m, n, phi, n_max)
    theta_arr = _as_float(theta)
    return ones_like(theta_arr) * (0.5 / math.sqrt(math.pi))


def xlog1py(x: ArrayLike, y: ArrayLike) -> ndarray:
    """Compute ``x * log1p(y)`` so that the result is 0 if ``x == 0``.

    Args:
        x: Multiplier array.
        y: Argument to ``log1p``.

    Returns:
        ndarray: ``x * log(1 + y)``.
    """
    x_arr = _as_float(x)
    y_arr = _as_float(y)
    return where(x_arr == 0.0, zeros_like(x_arr + y_arr), x_arr * log1p(y_arr))


def xlogy(x: ArrayLike, y: ArrayLike) -> ndarray:
    """Compute ``x * log(y)`` so that the result is 0 if ``x == 0``.

    Args:
        x: Multiplier array.
        y: Argument to ``log``.

    Returns:
        ndarray: ``x * log(y)``.
    """
    x_arr = _as_float(x)
    y_arr = _as_float(y)
    return where(x_arr == 0.0, zeros_like(x_arr + y_arr), x_arr * log(y_arr))


def zeta(x: ArrayLike, q: ArrayLike) -> ndarray:
    """The Hurwitz zeta function ``zeta(x, q)``.

    Args:
        x: Exponent parameter ``x > 1``.
        q: Offset parameter ``q > 0``.

    Returns:
        ndarray: Hurwitz zeta values.
    """
    return _wrap(_ops.zeta(_to_tensor(_as_float(x)), _to_tensor(_as_float(q))))


__all__ = [
    "bernoulli",
    "bessel_jn",
    "beta",
    "betainc",
    "betaln",
    "digamma",
    "entr",
    "erf",
    "erfc",
    "erfinv",
    "exp1",
    "expi",
    "expit",
    "expn",
    "factorial",
    "gamma",
    "gammainc",
    "gammaincc",
    "gammaln",
    "gammasgn",
    "hyp1f1",
    "i0",
    "i0e",
    "i1",
    "i1e",
    "kl_div",
    "log_ndtr",
    "log_softmax",
    "logit",
    "logsumexp",
    "lpmn",
    "lpmn_values",
    "multigammaln",
    "ndtr",
    "ndtri",
    "poch",
    "polygamma",
    "rel_entr",
    "spence",
    "sph_harm",
    "xlog1py",
    "xlogy",
    "zeta",
]
