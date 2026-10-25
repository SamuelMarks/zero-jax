"""Random number generation and probability distribution proxy operations."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.base import OpDef, get_op

ball: OpDef = get_op("Ball")()
bernoulli: OpDef = get_op("Bernoulli")()
beta: OpDef = get_op("Beta")()
binomial: OpDef = get_op("Binomial")()
bits: OpDef = get_op("Bits")()
cauchy: OpDef = get_op("Cauchy")()
chisquare: OpDef = get_op("Chisquare")()
clone: OpDef = get_op("Clone")()
dirichlet: OpDef = get_op("Dirichlet")()
double_sided_maxwell: OpDef = get_op("DoubleSidedMaxwell")()
exponential: OpDef = get_op("Exponential")()
f: OpDef = get_op("F")()
gamma: OpDef = get_op("Gamma")()
generalized_normal: OpDef = get_op("GeneralizedNormal")()
geometric: OpDef = get_op("Geometric")()
gumbel: OpDef = get_op("Gumbel")()
key: OpDef = get_op("Key")()
key_data: OpDef = get_op("KeyData")()
key_impl: OpDef = get_op("KeyImpl")()
laplace: OpDef = get_op("Laplace")()
loggamma: OpDef = get_op("Loggamma")()
lognormal: OpDef = get_op("Lognormal")()
maxwell: OpDef = get_op("Maxwell")()
multivariate_normal: OpDef = get_op("MultivariateNormal")()
orthogonal: OpDef = get_op("Orthogonal")()
pareto: OpDef = get_op("Pareto")()
poisson: OpDef = get_op("Poisson")()
random_gamma_grad: OpDef = get_op("RandomGammaGrad")()
rayleigh: OpDef = get_op("Rayleigh")()
t: OpDef = get_op("T")()
triangular: OpDef = get_op("Triangular")()
wald: OpDef = get_op("Wald")()
weibull_min: OpDef = get_op("WeibullMin")()
wrap_key_data: OpDef = get_op("WrapKeyData")()

__all__ = [
    "ball",
    "bernoulli",
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
    "lognormal",
    "maxwell",
    "multivariate_normal",
    "orthogonal",
    "pareto",
    "poisson",
    "random_gamma_grad",
    "rayleigh",
    "t",
    "triangular",
    "wald",
    "weibull_min",
    "wrap_key_data",
]
