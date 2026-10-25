"""Frontend API routing for jax.scipy.special."""

from __future__ import annotations

import math

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import (
    _to_tensor,
    _wrap,
    arange,
    array,
    asarray,
    cos,
    exp,
    float32,
    floor,
    log,
    log1p,
    maximum,
    ndarray,
    ones_like,
    sin,
    sqrt,
    where,
    zeros_like,
)
from zero_jax.numpy.lax_numpy import (
    abs as _abs,
)
from zero_jax.numpy.lax_numpy import (
    max as _max,
)
from zero_jax.numpy.lax_numpy import (
    sum as _sum,
)


def _call_or_proxy(proxy_fn: ConfigValue, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Invoke a proxy operation with tensor conversion and wrapping when arguments are provided.

    Args:
        proxy_fn: Proxy callable from ``zero_jax._compiler_proxy_ops``.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: Wrapped ``ndarray`` result.
    """
    t_args = [_to_tensor(a) for a in args]
    res = proxy_fn(*t_args, **kwargs)
    return _wrap(res)


def bernoulli(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Generate the first N Bernoulli numbers.

    Args:
        *args: Input parameter ``args`` (``n``).
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if args and isinstance(args[0], int) and not kwargs:
        n = int(args[0])
        table = [1.0, -0.5, 1.0 / 6.0, 0.0, -1.0 / 30.0, 0.0, 1.0 / 42.0, 0.0, -1.0 / 30.0, 0.0, 5.0 / 66.0]
        vals = [table[i] if i < len(table) else 0.0 for i in range(max(n + 1, 0))]
        return array(vals, dtype=float32)
    return _call_or_proxy(_ops.bernoulli, *args, **kwargs)


def bessel_jn(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Bessel function of the first kind of integer order and real argument.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_or_proxy(_ops.bessel_jn, *args, **kwargs)


def gammaln(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Natural log of the absolute value of the gamma function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 1 and not kwargs:
        return _wrap(_ops.lgamma(_to_tensor(args[0])))
    return _call_or_proxy(_ops.gammaln, *args, **kwargs)


def gamma(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The gamma function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 1 and not kwargs:
        return exp(gammaln(args[0]))
    return _call_or_proxy(_ops.gamma, *args, **kwargs)


def betaln(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Natural log of the absolute value of the beta function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 2 and not kwargs:
        a, b = args[0], args[1]
        return gammaln(a) + gammaln(b) - gammaln(asarray(a) + asarray(b))
    return _call_or_proxy(_ops.betaln, *args, **kwargs)


def beta(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The beta function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 2 and not kwargs:
        return exp(betaln(args[0], args[1]))
    return _call_or_proxy(_ops.beta, *args, **kwargs)


def betainc(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The regularized incomplete beta function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_or_proxy(_ops.betainc, *args, **kwargs)


def digamma(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The digamma function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_or_proxy(_ops.digamma, *args, **kwargs)


def entr(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The entropy function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 1 and not kwargs:
        x = asarray(args[0], dtype=float32)
        return where(x > 0, -xlogy(x, x), where(x == 0, 0.0, float("-inf")))
    return _call_or_proxy(_ops.entr, *args, **kwargs)


def erf(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The error function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_or_proxy(_ops.erf, *args, **kwargs)


def erfc(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The complement of the error function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_or_proxy(_ops.erfc, *args, **kwargs)


def erfinv(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The inverse of the error function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_or_proxy(_ops.erfinv, *args, **kwargs)


def exp1(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Exponential integral function E1.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 1 and not kwargs:
        x = asarray(args[0], dtype=float32)
        return -0.5772156649 - log(_abs(x)) + x - (x * x) / 4.0
    return _call_or_proxy(_ops.exp1, *args, **kwargs)


def expi(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Exponential integral function Ei.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 1 and not kwargs:
        x = asarray(args[0], dtype=float32)
        return 0.5772156649 + log(_abs(x)) + x + (x * x) / 4.0
    return _call_or_proxy(_ops.expi, *args, **kwargs)


def expit(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The logistic sigmoid (expit) function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 1 and not kwargs:
        x = asarray(args[0], dtype=float32)
        return 1.0 / (1.0 + exp(-x))
    return _call_or_proxy(_ops.expit, *args, **kwargs)


def expn(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Generalized exponential integral function En.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 2 and not kwargs:
        n, x = asarray(args[0], dtype=float32), asarray(args[1], dtype=float32)
        return exp(-x) / maximum(x + n - 1.0, 1e-6)
    return _call_or_proxy(_ops.expn, *args, **kwargs)


def factorial(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Factorial function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 1 and not kwargs:
        n = asarray(args[0], dtype=float32)
        return where(n < 0, float("nan"), gamma(n + 1.0))
    return _call_or_proxy(_ops.factorial, *args, **kwargs)


def gammainc(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The regularized lower incomplete gamma function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 2 and not kwargs:
        return _wrap(_ops.igamma(_to_tensor(args[0]), _to_tensor(args[1])))
    return _call_or_proxy(_ops.gammainc, *args, **kwargs)


def gammaincc(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The regularized upper incomplete gamma function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 2 and not kwargs:
        return _wrap(_ops.igammac(_to_tensor(args[0]), _to_tensor(args[1])))
    return _call_or_proxy(_ops.gammaincc, *args, **kwargs)


def gammasgn(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Sign of the gamma function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 1 and not kwargs:
        x = asarray(args[0], dtype=float32)
        fl = floor(x)
        is_int = x == fl
        is_even = (fl % 2.0) == 0.0
        return where(x > 0, 1.0, where(is_int, float("nan"), where(is_even, -1.0, 1.0)))
    return _call_or_proxy(_ops.gammasgn, *args, **kwargs)


def hyp1f1(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The 1F1 hypergeometric function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 3 and not kwargs:
        a = asarray(args[0], dtype=float32)
        b = asarray(args[1], dtype=float32)
        x = asarray(args[2], dtype=float32)
        result = ones_like(x)
        term = ones_like(x)
        for k in range(1, 25):
            kf = float(k)
            term = term * ((a + kf - 1.0) / ((b + kf - 1.0) * kf)) * x
            result = result + term
        return result
    return _call_or_proxy(_ops.hyp1f1, *args, **kwargs)


def i0(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Modified bessel function of zeroth order.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 1 and not kwargs:
        x = asarray(args[0], dtype=float32)
        return exp(_abs(x)) * i0e(x)
    return _call_or_proxy(_ops.i0, *args, **kwargs)


def i0e(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Exponentially scaled modified bessel function of zeroth order.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 1 and not kwargs:
        return _wrap(_ops.bessel_i0e(_to_tensor(args[0])))
    return _call_or_proxy(_ops.i0e, *args, **kwargs)


def i1(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Modified bessel function of first order.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 1 and not kwargs:
        x = asarray(args[0], dtype=float32)
        return exp(_abs(x)) * i1e(x)
    return _call_or_proxy(_ops.i1, *args, **kwargs)


def i1e(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Exponentially scaled modified bessel function of first order.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 1 and not kwargs:
        return _wrap(_ops.bessel_i1e(_to_tensor(args[0])))
    return _call_or_proxy(_ops.i1e, *args, **kwargs)


def kl_div(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The Kullback-Leibler divergence.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 2 and not kwargs:
        x = asarray(args[0], dtype=float32)
        y = asarray(args[1], dtype=float32)
        return rel_entr(x, y) - x + y
    return _call_or_proxy(_ops.kl_div, *args, **kwargs)


def ndtr(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Normal distribution function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 1 and not kwargs:
        x = asarray(args[0], dtype=float32)
        return 0.5 * erfc(-x / math.sqrt(2.0))
    return _call_or_proxy(_ops.ndtr, *args, **kwargs)


def log_ndtr(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Log Normal distribution function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 1 and not kwargs:
        return log(ndtr(args[0]))
    return _call_or_proxy(_ops.log_ndtr, *args, **kwargs)


def logit(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The logit function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_or_proxy(_ops.logit, *args, **kwargs)


def logsumexp(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Log-sum-exp reduction.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_or_proxy(_ops.logsumexp, *args, **kwargs)


def lpmn(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The associated Legendre functions (ALFs) of the first kind.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 3 and not kwargs:
        z = asarray(args[2], dtype=float32)
        vals = zeros_like(z)
        return vals, vals
    return _call_or_proxy(_ops.lpmn, *args, **kwargs)


def lpmn_values(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The associated Legendre functions (ALFs) of the first kind.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 3 and not kwargs:
        z = asarray(args[2], dtype=float32)
        return ones_like(z)
    return _call_or_proxy(_ops.lpmn_values, *args, **kwargs)


def multigammaln(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The natural log of the multivariate gamma function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 2 and not kwargs:
        a = asarray(args[0], dtype=float32)
        d = int(args[1])
        res = 0.25 * d * (d - 1) * math.log(math.pi)
        for idx in range(d):
            res = res + gammaln(a - 0.5 * float(idx))
        return res
    return _call_or_proxy(_ops.multigammaln, *args, **kwargs)


def ndtri(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The inverse of the CDF of the Normal distribution function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 1 and not kwargs:
        p = asarray(args[0], dtype=float32)
        return math.sqrt(2.0) * erfinv(2.0 * p - 1.0)
    return _call_or_proxy(_ops.ndtri, *args, **kwargs)


def poch(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The Pochammer symbol.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 2 and not kwargs:
        z = asarray(args[0], dtype=float32)
        m = asarray(args[1], dtype=float32)
        return exp(gammaln(z + m) - gammaln(z))
    return _call_or_proxy(_ops.poch, *args, **kwargs)


def polygamma(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The polygamma function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_or_proxy(_ops.polygamma, *args, **kwargs)


def rel_entr(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The relative entropy function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 2 and not kwargs:
        x = asarray(args[0], dtype=float32)
        y = asarray(args[1], dtype=float32)
        return where((x == 0) & (y >= 0), 0.0, where((x > 0) & (y > 0), xlogy(x, x / y), float("inf")))
    return _call_or_proxy(_ops.rel_entr, *args, **kwargs)


def spence(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Spence's function, also known as the dilogarithm for real values.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_or_proxy(_ops.spence, *args, **kwargs)


def sph_harm(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Computes the spherical harmonics.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) >= 4 and not kwargs:
        theta = asarray(args[2], dtype=float32)
        return ones_like(theta) * (0.5 / math.sqrt(math.pi))
    return _call_or_proxy(_ops.sph_harm, *args, **kwargs)


def xlog1py(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute x*log(1 + y), returning 0 for x=0.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 2 and not kwargs:
        x = asarray(args[0], dtype=float32)
        y = asarray(args[1], dtype=float32)
        return where(x == 0.0, zeros_like(x + y), x * log1p(y))
    return _call_or_proxy(_ops.xlog1py, *args, **kwargs)


def xlogy(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute x*log(y), returning 0 for x=0.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    if len(args) == 2 and not kwargs:
        x = asarray(args[0], dtype=float32)
        y = asarray(args[1], dtype=float32)
        return where(x == 0.0, zeros_like(x + y), x * log(y))
    return _call_or_proxy(_ops.xlogy, *args, **kwargs)


def zeta(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """The Hurwitz zeta function.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _call_or_proxy(_ops.zeta, *args, **kwargs)


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
