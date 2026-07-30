"""Frontend API routing for jax.scipy.special."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def bernoulli(*args: Any, **kwargs: Any) -> Any:
    """Generate the first N Bernoulli numbers."""
    return _ops.bernoulli(*args, **kwargs)


def bessel_jn(*args: Any, **kwargs: Any) -> Any:
    """Bessel function of the first kind of integer order and real argument."""
    return _ops.bessel_jn(*args, **kwargs)


def beta(*args: Any, **kwargs: Any) -> Any:
    """The beta function"""
    return _ops.beta(*args, **kwargs)


def betainc(*args: Any, **kwargs: Any) -> Any:
    """The regularized incomplete beta function."""
    return _ops.betainc(*args, **kwargs)


def betaln(*args: Any, **kwargs: Any) -> Any:
    """Natural log of the absolute value of the beta function"""
    return _ops.betaln(*args, **kwargs)


def digamma(*args: Any, **kwargs: Any) -> Any:
    """The digamma function"""
    return _ops.digamma(*args, **kwargs)


def entr(*args: Any, **kwargs: Any) -> Any:
    """The entropy function"""
    return _ops.entr(*args, **kwargs)


def erf(*args: Any, **kwargs: Any) -> Any:
    """The error function"""
    return _ops.erf(*args, **kwargs)


def erfc(*args: Any, **kwargs: Any) -> Any:
    """The complement of the error function"""
    return _ops.erfc(*args, **kwargs)


def erfinv(*args: Any, **kwargs: Any) -> Any:
    """The inverse of the error function"""
    return _ops.erfinv(*args, **kwargs)


def exp1(*args: Any, **kwargs: Any) -> Any:
    """Exponential integral function."""
    return _ops.exp1(*args, **kwargs)


def expi(*args: Any, **kwargs: Any) -> Any:
    """Exponential integral function."""
    return _ops.expi(*args, **kwargs)


def expit(*args: Any, **kwargs: Any) -> Any:
    """The logistic sigmoid (expit) function"""
    return _ops.expit(*args, **kwargs)


def expn(*args: Any, **kwargs: Any) -> Any:
    """Generalized exponential integral function."""
    return _ops.expn(*args, **kwargs)


def factorial(*args: Any, **kwargs: Any) -> Any:
    """Factorial function"""
    return _ops.factorial(*args, **kwargs)


def gamma(*args: Any, **kwargs: Any) -> Any:
    """The gamma function."""
    return _ops.gamma(*args, **kwargs)


def gammainc(*args: Any, **kwargs: Any) -> Any:
    """The regularized lower incomplete gamma function."""
    return _ops.gammainc(*args, **kwargs)


def gammaincc(*args: Any, **kwargs: Any) -> Any:
    """The regularized upper incomplete gamma function."""
    return _ops.gammaincc(*args, **kwargs)


def gammaln(*args: Any, **kwargs: Any) -> Any:
    """Natural log of the absolute value of the gamma function."""
    return _ops.gammaln(*args, **kwargs)


def gammasgn(*args: Any, **kwargs: Any) -> Any:
    """Sign of the gamma function."""
    return _ops.gammasgn(*args, **kwargs)


def hyp1f1(*args: Any, **kwargs: Any) -> Any:
    """The 1F1 hypergeometric function."""
    return _ops.hyp1f1(*args, **kwargs)


def i0(*args: Any, **kwargs: Any) -> Any:
    """Modified bessel function of zeroth order."""
    return _ops.i0(*args, **kwargs)


def i0e(*args: Any, **kwargs: Any) -> Any:
    """Exponentially scaled modified bessel function of zeroth order."""
    return _ops.i0e(*args, **kwargs)


def i1(*args: Any, **kwargs: Any) -> Any:
    """Modified bessel function of first order."""
    return _ops.i1(*args, **kwargs)


def i1e(*args: Any, **kwargs: Any) -> Any:
    """Exponentially scaled modified bessel function of first order."""
    return _ops.i1e(*args, **kwargs)


def kl_div(*args: Any, **kwargs: Any) -> Any:
    """The Kullback-Leibler divergence."""
    return _ops.kl_div(*args, **kwargs)


def log_ndtr(*args: Any, **kwargs: Any) -> Any:
    """Log Normal distribution function."""
    return _ops.log_ndtr(*args, **kwargs)


def logit(*args: Any, **kwargs: Any) -> Any:
    """The logit function"""
    return _ops.logit(*args, **kwargs)


def logsumexp(*args: Any, **kwargs: Any) -> Any:
    """Log-sum-exp reduction."""
    return _ops.logsumexp(*args, **kwargs)


def lpmn(*args: Any, **kwargs: Any) -> Any:
    """The associated Legendre functions (ALFs) of the first kind."""
    return _ops.lpmn(*args, **kwargs)


def lpmn_values(*args: Any, **kwargs: Any) -> Any:
    """The associated Legendre functions (ALFs) of the first kind."""
    return _ops.lpmn_values(*args, **kwargs)


def multigammaln(*args: Any, **kwargs: Any) -> Any:
    """The natural log of the multivariate gamma function."""
    return _ops.multigammaln(*args, **kwargs)


def ndtr(*args: Any, **kwargs: Any) -> Any:
    """Normal distribution function."""
    return _ops.ndtr(*args, **kwargs)


def ndtri(*args: Any, **kwargs: Any) -> Any:
    """The inverse of the CDF of the Normal distribution function."""
    return _ops.ndtri(*args, **kwargs)


def poch(*args: Any, **kwargs: Any) -> Any:
    """The Pochammer symbol."""
    return _ops.poch(*args, **kwargs)


def polygamma(*args: Any, **kwargs: Any) -> Any:
    """The polygamma function."""
    return _ops.polygamma(*args, **kwargs)


def rel_entr(*args: Any, **kwargs: Any) -> Any:
    """The relative entropy function."""
    return _ops.rel_entr(*args, **kwargs)


def spence(*args: Any, **kwargs: Any) -> Any:
    """Spence's function, also known as the dilogarithm for real values."""
    return _ops.spence(*args, **kwargs)


def sph_harm(*args: Any, **kwargs: Any) -> Any:
    """Computes the spherical harmonics."""
    return _ops.sph_harm(*args, **kwargs)


def xlog1py(*args: Any, **kwargs: Any) -> Any:
    """Compute x*log(1 + y), returning 0 for x=0."""
    return _ops.xlog1py(*args, **kwargs)


def xlogy(*args: Any, **kwargs: Any) -> Any:
    """Compute x*log(y), returning 0 for x=0."""
    return _ops.xlogy(*args, **kwargs)


def zeta(*args: Any, **kwargs: Any) -> Any:
    """The Hurwitz zeta function."""
    return _ops.zeta(*args, **kwargs)
