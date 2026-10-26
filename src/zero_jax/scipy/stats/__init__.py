"""Frontend API routing for jax.scipy.stats."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec, ConfigValue
from zero_jax.numpy.lax_numpy import argmax, argsort, asarray, exp, float32, mean, pi, sqrt, std, unique, var

from . import (
    bernoulli,
    beta,
    betabinom,
    binom,
    cauchy,
    chi2,
    dirichlet,
    expon,
    gamma,
    gennorm,
    geom,
    laplace,
    logistic,
    multinomial,
    multivariate_normal,
    nbinom,
    norm,
    pareto,
    poisson,
    t,
    truncnorm,
    uniform,
    vonmises,
    wrapcauchy,
)


class gaussian_kde:
    """Gaussian Kernel Density Estimator."""

    def __init__(
        self,
        dataset: ArrayLikeSpec = None,
        bw_method: ConfigValue = None,
        weights: ArrayLikeSpec = None,
        *,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize ``gaussian_kde`` estimator.

        Args:
            dataset: Training data points.
            bw_method: Bandwidth estimation method or scalar factor.
            weights: Optional weights for data points.
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
        """
        self.dataset = dataset
        self.bw_method = bw_method
        self.weights = weights
        self.id = id
        self.name = name
        self.value = value

    def __call__(self, points: ArrayLikeSpec) -> ArrayLikeSpec:
        """Evaluate the estimated density on ``points``.

        Args:
            points: Query points.

        Returns:
            ArrayLikeSpec: Evaluated density values.
        """
        return self.evaluate(points)

    def evaluate(self, points: ArrayLikeSpec) -> ArrayLikeSpec:
        """Evaluate the estimated density on ``points``.

        Args:
            points: Query points.

        Returns:
            ArrayLikeSpec: Evaluated density values.
        """
        pts = asarray(points, dtype=float32)
        if self.dataset is None:
            return pts
        data = asarray(self.dataset, dtype=float32)
        bw = sqrt(var(data) + 1e-6)
        diff = pts[..., None] - data.ravel()
        return mean(exp(-0.5 * (diff / bw) ** 2) / (bw * sqrt(2.0 * pi)), axis=-1)


def mode(a: ArrayLikeSpec, axis: int | None = 0, nan_policy: str = "propagate", keepdims: bool = False) -> tuple[ArrayLikeSpec, ArrayLikeSpec]:
    """Compute the mode (most common value) and its count along an axis.

    Args:
        a: Input array.
        axis: Axis along which to compute the mode.
        nan_policy: Policy for handling NaNs.
        keepdims: Whether to retain reduced dimensions.

    Returns:
        tuple[ArrayLikeSpec, ArrayLikeSpec]: Tuple of ``(mode_val, count)``.
    """
    _ = (axis, nan_policy, keepdims)
    vals, counts = unique(asarray(a), return_counts=True)
    idx = argmax(counts)
    return vals[idx], counts[idx]


def rankdata(
    a: ArrayLikeSpec,
    method: str = "average",
    *,
    axis: int | None = None,
    nan_policy: str = "propagate",
) -> ArrayLikeSpec:
    """Assign ranks to data, dealing with ties appropriately.

    Args:
        a: Input array of values to be ranked.
        method: Tie-breaking method name.
        axis: Optional axis along which to rank.
        nan_policy: Policy for handling NaNs.

    Returns:
        ArrayLikeSpec: Array of 1-based ranks.
    """
    _ = (method, axis, nan_policy)
    arr = asarray(a)
    order = argsort(argsort(arr))
    return asarray(order, dtype=float32) + 1.0


def sem(
    a: ArrayLikeSpec,
    axis: int | None = 0,
    ddof: int = 1,
    nan_policy: str = "propagate",
    *,
    keepdims: bool = False,
) -> ArrayLikeSpec:
    """Compute the standard error of the mean.

    Args:
        a: Input array.
        axis: Axis along which to operate.
        ddof: Delta degrees of freedom.
        nan_policy: Policy for handling NaNs.
        keepdims: Whether to retain reduced dimensions.

    Returns:
        ArrayLikeSpec: Standard error of the mean.
    """
    _ = (axis, nan_policy, keepdims)
    arr = asarray(a, dtype=float32)
    return std(arr, ddof=ddof) / sqrt(float(max(arr.size, 1)))


__all__ = [
    "bernoulli",
    "beta",
    "betabinom",
    "binom",
    "cauchy",
    "chi2",
    "dirichlet",
    "expon",
    "gamma",
    "gaussian_kde",
    "gennorm",
    "geom",
    "laplace",
    "logistic",
    "mode",
    "multinomial",
    "multivariate_normal",
    "nbinom",
    "norm",
    "pareto",
    "poisson",
    "rankdata",
    "sem",
    "t",
    "truncnorm",
    "uniform",
    "vonmises",
    "wrapcauchy",
]
