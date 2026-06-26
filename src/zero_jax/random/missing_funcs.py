"""Missing random distribution functions."""

from __future__ import annotations
from typing import Any


def ball(key: Any, d: int, p: float = 2.0, shape: Any = ()) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.ball(_to_tensor(key), d, _to_tensor(p), shape))


def beta(key: Any, a: Any, b: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.beta(_to_tensor(key), _to_tensor(a), _to_tensor(b), shape, dtype))


def binomial(key: Any, n: Any, p: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.binomial(_to_tensor(key), _to_tensor(n), _to_tensor(p), shape, dtype)
    )


def bits(key: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.bits(_to_tensor(key), shape, dtype))


def cauchy(key: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.cauchy(_to_tensor(key), shape, dtype))


def chisquare(key: Any, df: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.chisquare(_to_tensor(key), _to_tensor(df), shape, dtype))


def clone(key: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.clone(_to_tensor(key)))


def dirichlet(key: Any, alpha: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.dirichlet(_to_tensor(key), _to_tensor(alpha), shape, dtype))


def double_sided_maxwell(
    key: Any, loc: Any, scale: Any, shape: Any = None, dtype: Any = None
) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.double_sided_maxwell(
            _to_tensor(key), _to_tensor(loc), _to_tensor(scale), shape, dtype
        )
    )


def exponential(key: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.exponential(_to_tensor(key), shape, dtype))


def f(key: Any, dfnum: Any, dfden: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.f(_to_tensor(key), _to_tensor(dfnum), _to_tensor(dfden), shape, dtype)
    )


def gamma(key: Any, a: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.gamma(_to_tensor(key), _to_tensor(a), shape, dtype))


def generalized_normal(key: Any, p: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.generalized_normal(_to_tensor(key), _to_tensor(p), shape, dtype))


def geometric(key: Any, p: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.geometric(_to_tensor(key), _to_tensor(p), shape, dtype))


def gumbel(key: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.gumbel(_to_tensor(key), shape, dtype))


def key(seed: int) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.key(seed))


def key_data(k: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.key_data(k))


def key_impl(k: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.key_impl(k))


def laplace(key: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.laplace(_to_tensor(key), shape, dtype))


def loggamma(key: Any, a: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.loggamma(_to_tensor(key), _to_tensor(a), shape, dtype))


def logistic(key: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.logistic(_to_tensor(key), shape, dtype))


def lognormal(key: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.lognormal(_to_tensor(key), shape, dtype))


def maxwell(key: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.maxwell(_to_tensor(key), shape, dtype))


def multivariate_normal(
    key: Any,
    mean: Any,
    cov: Any,
    shape: Any = None,
    dtype: Any = None,
    method: str = "svd",
) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.multivariate_normal(
            _to_tensor(key), _to_tensor(mean), _to_tensor(cov), shape, dtype, method
        )
    )


def orthogonal(key: Any, n: int, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.orthogonal(_to_tensor(key), n, shape, dtype))


def pareto(key: Any, b: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.pareto(_to_tensor(key), _to_tensor(b), shape, dtype))


def poisson(key: Any, lam: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.poisson(_to_tensor(key), _to_tensor(lam), shape, dtype))


def rademacher(key: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.rademacher(_to_tensor(key), shape, dtype))


def random_gamma_p() -> Any:
    pass


def rayleigh(key: Any, scale: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.rayleigh(_to_tensor(key), _to_tensor(scale), shape, dtype))


def t(key: Any, df: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(ops.t(_to_tensor(key), _to_tensor(df), shape, dtype))


def triangular(
    key: Any, left: Any, mode: Any, right: Any, shape: Any = None, dtype: Any = None
) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

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


def wald(key: Any, mean: Any, scale: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.wald(_to_tensor(key), _to_tensor(mean), _to_tensor(scale), shape, dtype)
    )


def weibull_min(
    key: Any, scale: Any, concentration: Any, shape: Any = None, dtype: Any = None
) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

    return _wrap(
        ops.weibull_min(
            _to_tensor(key), _to_tensor(scale), _to_tensor(concentration), shape, dtype
        )
    )


def wrap_key_data(key_data: Any) -> Any:
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap
    import ml_switcheroo_compiler.ops as ops

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
