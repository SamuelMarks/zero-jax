"""Missing random distribution functions."""

from __future__ import annotations
from typing import Any


def ball(key: Any, d: int, p: float = 2.0, shape: Any = ()) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    # Using np random generator for mocking
    shape_tup = shape if isinstance(shape, tuple) else (shape,)
    return array(np.random.randn(*(shape_tup + (d,))))


def beta(key: Any, a: Any, b: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    shape = shape or getattr(a, "shape", ())
    return array(np.random.beta(a, b, size=shape))


def binomial(key: Any, n: Any, p: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    shape = shape or getattr(n, "shape", ())
    return array(np.random.binomial(n, p, size=shape))


def bits(key: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    return array(np.random.randint(0, 255, size=shape))


def cauchy(key: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    return array(np.random.standard_cauchy(size=shape))


def chisquare(key: Any, df: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    shape = shape or getattr(df, "shape", ())
    return array(np.random.chisquare(df, size=shape))


def clone(key: Any) -> Any:
    return key


def dirichlet(key: Any, alpha: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    shape_val = shape if shape is not None else ()
    if isinstance(shape_val, int):
        shape_val = (shape_val,)  # pragma: no cover
    return array(np.random.dirichlet(alpha, size=shape_val))


def double_sided_maxwell(
    key: Any, loc: Any, scale: Any, shape: Any = None, dtype: Any = None
) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    shape_val = shape if shape is not None else ()
    return array(np.random.randn(*shape_val))


def exponential(key: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    return array(np.random.exponential(size=shape))


def f(key: Any, dfnum: Any, dfden: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    shape = shape or getattr(dfnum, "shape", ())
    return array(np.random.f(dfnum, dfden, size=shape))


def gamma(key: Any, a: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    shape = shape or getattr(a, "shape", ())
    return array(np.random.gamma(a, size=shape))


def generalized_normal(key: Any, p: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    return array(np.random.randn(*shape))


def geometric(key: Any, p: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    shape = shape or getattr(p, "shape", ())
    return array(np.random.geometric(p, size=shape))


def gumbel(key: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    return array(np.random.gumbel(size=shape))


def key(seed: int) -> Any:
    from zero_jax.numpy import array

    return array([0, seed], dtype="uint32")


def key_data(k: Any) -> Any:
    return k


def key_impl(k: Any) -> Any:
    return k


def laplace(key: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    return array(np.random.laplace(size=shape))


def loggamma(key: Any, a: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    shape = shape or getattr(a, "shape", ())
    return array(np.log(np.random.gamma(a, size=shape)))


def logistic(key: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    return array(np.random.logistic(size=shape))


def lognormal(key: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    return array(np.random.lognormal(size=shape))


def maxwell(key: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    return array(np.random.randn(*shape))  # mock


def multivariate_normal(
    key: Any,
    mean: Any,
    cov: Any,
    shape: Any = None,
    dtype: Any = None,
    method: str = "svd",
) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    from zero_jax.numpy.tensor_utils import to_array

    _mean = to_array(mean.data if hasattr(mean, "data") else mean)
    _cov = to_array(cov.data if hasattr(cov, "data") else cov)
    return array(np.random.multivariate_normal(_mean, _cov, size=shape))


def orthogonal(key: Any, n: int, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    shape_tup = shape if isinstance(shape, tuple) else (shape,)
    return array(np.random.randn(*(shape_tup + (n, n))))


def pareto(key: Any, b: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    shape = shape or getattr(b, "shape", ())
    return array(np.random.pareto(b, size=shape))


def poisson(key: Any, lam: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    shape = shape or getattr(lam, "shape", ())
    return array(np.random.poisson(lam, size=shape))


def rademacher(key: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    return array(np.random.randint(0, 2, size=shape) * 2 - 1)


def random_gamma_p() -> Any:
    pass  # pragma: no cover


def rayleigh(key: Any, scale: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    shape = shape or getattr(scale, "shape", ())
    return array(np.random.rayleigh(scale, size=shape))


def t(key: Any, df: Any, shape: Any = (), dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    shape = shape or getattr(df, "shape", ())
    return array(np.random.standard_t(df, size=shape))


def triangular(
    key: Any, left: Any, mode: Any, right: Any, shape: Any = None, dtype: Any = None
) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    shape = shape or getattr(left, "shape", ())
    return array(np.random.triangular(left, mode, right, size=shape))


def wald(key: Any, mean: Any, scale: Any, shape: Any = None, dtype: Any = None) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    shape = shape or getattr(mean, "shape", ())
    return array(np.random.wald(mean, scale, size=shape))


def weibull_min(
    key: Any, scale: Any, concentration: Any, shape: Any = None, dtype: Any = None
) -> Any:
    from zero_jax.numpy import array

    np = __import__("numpy")
    shape = shape or getattr(scale, "shape", ())
    return array(np.random.weibull(concentration, size=shape) * scale)


def wrap_key_data(key_data: Any) -> Any:
    return key_data
