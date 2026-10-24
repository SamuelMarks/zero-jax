"""Missing random distribution functions."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec


def ball(key: ArrayLikeSpec, d: int, p: float = 2.0, shape: ArrayLikeSpec = ()) -> ArrayLikeSpec:
    """Execute ``ball``.

    Args:
        key: Input parameter ``key``.
        d: Input parameter ``d``.
        p: Input parameter ``p``.
        shape: Input parameter ``shape``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.ball(_to_tensor(key), d, _to_tensor(p), shape))


def beta(key: ArrayLikeSpec, a: ArrayLikeSpec, b: ArrayLikeSpec, shape: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``beta``.

    Args:
        key: Input parameter ``key``.
        a: Input parameter ``a``.
        b: Input parameter ``b``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.beta(_to_tensor(key), _to_tensor(a), _to_tensor(b), shape, dtype))


def binomial(key: ArrayLikeSpec, n: ArrayLikeSpec, p: ArrayLikeSpec, shape: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``binomial``.

    Args:
        key: Input parameter ``key``.
        n: Input parameter ``n``.
        p: Input parameter ``p``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.binomial(_to_tensor(key), _to_tensor(n), _to_tensor(p), shape, dtype))


def bits(key: ArrayLikeSpec, shape: ArrayLikeSpec = (), dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``bits``.

    Args:
        key: Input parameter ``key``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.bits(_to_tensor(key), shape, dtype))


def cauchy(key: ArrayLikeSpec, shape: ArrayLikeSpec = (), dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``cauchy``.

    Args:
        key: Input parameter ``key``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.cauchy(_to_tensor(key), shape, dtype))


def chisquare(key: ArrayLikeSpec, df: ArrayLikeSpec, shape: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``chisquare``.

    Args:
        key: Input parameter ``key``.
        df: Input parameter ``df``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.chisquare(_to_tensor(key), _to_tensor(df), shape, dtype))


def clone(key: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``clone``.

    Args:
        key: Input parameter ``key``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.clone(_to_tensor(key)))


def dirichlet(key: ArrayLikeSpec, alpha: ArrayLikeSpec, shape: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``dirichlet``.

    Args:
        key: Input parameter ``key``.
        alpha: Input parameter ``alpha``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.dirichlet(_to_tensor(key), _to_tensor(alpha), shape, dtype))


def double_sided_maxwell(key: ArrayLikeSpec, loc: ArrayLikeSpec, scale: ArrayLikeSpec, shape: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``double_sided_maxwell``.

    Args:
        key: Input parameter ``key``.
        loc: Input parameter ``loc``.
        scale: Input parameter ``scale``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.double_sided_maxwell(_to_tensor(key), _to_tensor(loc), _to_tensor(scale), shape, dtype))


def exponential(key: ArrayLikeSpec, shape: ArrayLikeSpec = (), dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``exponential``.

    Args:
        key: Input parameter ``key``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.exponential(_to_tensor(key), shape, dtype))


def f(key: ArrayLikeSpec, dfnum: ArrayLikeSpec, dfden: ArrayLikeSpec, shape: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``f``.

    Args:
        key: Input parameter ``key``.
        dfnum: Input parameter ``dfnum``.
        dfden: Input parameter ``dfden``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.f(_to_tensor(key), _to_tensor(dfnum), _to_tensor(dfden), shape, dtype))


def gamma(key: ArrayLikeSpec, a: ArrayLikeSpec, shape: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``gamma``.

    Args:
        key: Input parameter ``key``.
        a: Input parameter ``a``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.gamma(_to_tensor(key), _to_tensor(a), shape, dtype))


def generalized_normal(key: ArrayLikeSpec, p: ArrayLikeSpec, shape: ArrayLikeSpec = (), dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``generalized_normal``.

    Args:
        key: Input parameter ``key``.
        p: Input parameter ``p``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.generalized_normal(_to_tensor(key), _to_tensor(p), shape, dtype))


def geometric(key: ArrayLikeSpec, p: ArrayLikeSpec, shape: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``geometric``.

    Args:
        key: Input parameter ``key``.
        p: Input parameter ``p``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.geometric(_to_tensor(key), _to_tensor(p), shape, dtype))


def gumbel(key: ArrayLikeSpec, shape: ArrayLikeSpec = (), dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``gumbel``.

    Args:
        key: Input parameter ``key``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.gumbel(_to_tensor(key), shape, dtype))


def key(seed: int) -> ArrayLikeSpec:
    """Execute ``key``.

    Args:
        seed: Input parameter ``seed``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _wrap

    return _wrap(ops.key(seed))


def key_data(k: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``key_data``.

    Args:
        k: Input parameter ``k``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _wrap

    return _wrap(ops.key_data(k))


def key_impl(k: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``key_impl``.

    Args:
        k: Input parameter ``k``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _wrap

    return _wrap(ops.key_impl(k))


def laplace(key: ArrayLikeSpec, shape: ArrayLikeSpec = (), dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``laplace``.

    Args:
        key: Input parameter ``key``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.laplace(_to_tensor(key), shape, dtype))


def loggamma(key: ArrayLikeSpec, a: ArrayLikeSpec, shape: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``loggamma``.

    Args:
        key: Input parameter ``key``.
        a: Input parameter ``a``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.loggamma(_to_tensor(key), _to_tensor(a), shape, dtype))


def logistic(key: ArrayLikeSpec, shape: ArrayLikeSpec = (), dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``logistic``.

    Args:
        key: Input parameter ``key``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.logistic(_to_tensor(key), shape, dtype))


def lognormal(key: ArrayLikeSpec, shape: ArrayLikeSpec = (), dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``lognormal``.

    Args:
        key: Input parameter ``key``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.lognormal(_to_tensor(key), shape, dtype))


def maxwell(key: ArrayLikeSpec, shape: ArrayLikeSpec = (), dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``maxwell``.

    Args:
        key: Input parameter ``key``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.maxwell(_to_tensor(key), shape, dtype))


def multivariate_normal(
    key: ArrayLikeSpec,
    mean: ArrayLikeSpec,
    cov: ArrayLikeSpec,
    shape: ArrayLikeSpec = None,
    dtype: ArrayLikeSpec = None,
    method: str = "svd",
) -> ArrayLikeSpec:
    """Execute ``multivariate_normal``.

    Args:
        key: Input parameter ``key``.
        mean: Input parameter ``mean``.
        cov: Input parameter ``cov``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.
        method: Input parameter ``method``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.multivariate_normal(_to_tensor(key), _to_tensor(mean), _to_tensor(cov), shape, dtype, method))


def orthogonal(key: ArrayLikeSpec, n: int, shape: ArrayLikeSpec = (), dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``orthogonal``.

    Args:
        key: Input parameter ``key``.
        n: Input parameter ``n``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.orthogonal(_to_tensor(key), n, shape, dtype))


def pareto(key: ArrayLikeSpec, b: ArrayLikeSpec, shape: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``pareto``.

    Args:
        key: Input parameter ``key``.
        b: Input parameter ``b``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.pareto(_to_tensor(key), _to_tensor(b), shape, dtype))


def poisson(key: ArrayLikeSpec, lam: ArrayLikeSpec, shape: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``poisson``.

    Args:
        key: Input parameter ``key``.
        lam: Input parameter ``lam``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.poisson(_to_tensor(key), _to_tensor(lam), shape, dtype))


def rademacher(key: ArrayLikeSpec, shape: ArrayLikeSpec = (), dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``rademacher``.

    Args:
        key: Input parameter ``key``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.rademacher(_to_tensor(key), shape, dtype))


def rayleigh(key: ArrayLikeSpec, scale: ArrayLikeSpec, shape: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``rayleigh``.

    Args:
        key: Input parameter ``key``.
        scale: Input parameter ``scale``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.rayleigh(_to_tensor(key), _to_tensor(scale), shape, dtype))


def t(key: ArrayLikeSpec, df: ArrayLikeSpec, shape: ArrayLikeSpec = (), dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``t``.

    Args:
        key: Input parameter ``key``.
        df: Input parameter ``df``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.t(_to_tensor(key), _to_tensor(df), shape, dtype))


def triangular(key: ArrayLikeSpec, left: ArrayLikeSpec, mode: ArrayLikeSpec, right: ArrayLikeSpec, shape: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``triangular``.

    Args:
        key: Input parameter ``key``.
        left: Input parameter ``left``.
        mode: Input parameter ``mode``.
        right: Input parameter ``right``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(
        ops.triangular(
            _to_tensor(key),
            _to_tensor(left),
            _to_tensor(mode),
            _to_tensor(right),
            shape,
            dtype,
        )
    )


def wald(key: ArrayLikeSpec, mean: ArrayLikeSpec, scale: ArrayLikeSpec, shape: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``wald``.

    Args:
        key: Input parameter ``key``.
        mean: Input parameter ``mean``.
        scale: Input parameter ``scale``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.wald(_to_tensor(key), _to_tensor(mean), _to_tensor(scale), shape, dtype))


def weibull_min(key: ArrayLikeSpec, scale: ArrayLikeSpec, concentration: ArrayLikeSpec, shape: ArrayLikeSpec = None, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``weibull_min``.

    Args:
        key: Input parameter ``key``.
        scale: Input parameter ``scale``.
        concentration: Input parameter ``concentration``.
        shape: Input parameter ``shape``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.weibull_min(_to_tensor(key), _to_tensor(scale), _to_tensor(concentration), shape, dtype))


def wrap_key_data(key_data: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``wrap_key_data``.

    Args:
        key_data: Input parameter ``key_data``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.wrap_key_data(_to_tensor(key_data)))


__all__ = [
    "ball",
    "beta",
    "binomial",
    "bits",
    "cauchy",
    "chisquare",
    "clone",
    "dirichlet",
    "double_sided_maxwell",
    "exponential",
    "f",
    "gamma",
    "generalized_normal",
    "geometric",
    "gumbel",
    "key",
    "key_data",
    "key_impl",
    "laplace",
    "loggamma",
    "logistic",
    "lognormal",
    "maxwell",
    "multivariate_normal",
    "orthogonal",
    "pareto",
    "poisson",
    "rademacher",
    "rayleigh",
    "t",
    "triangular",
    "wald",
    "weibull_min",
    "wrap_key_data",
]
