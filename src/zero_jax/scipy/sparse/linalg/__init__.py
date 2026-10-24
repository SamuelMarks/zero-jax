"""Frontend API routing for jax.scipy.sparse.linalg."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ConfigValue


def bicgstab(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Use Bi-Conjugate Gradient Stable iteration to solve ``Ax = b``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.bicgstab(*args, **kwargs)


def cg(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Use Conjugate Gradient iteration to solve ``Ax = b``.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.cg(*args, **kwargs)


def gmres(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """GMRES solves the linear system A x = b for x, given A and b.

    Args:
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return _ops.gmres(*args, **kwargs)


class LinearOperator:
    """Frontend class for ``LinearOperator``."""


class LinearOperatorAdjoint(LinearOperator):
    """Frontend class for ``LinearOperatorAdjoint``."""


class LinearOperatorBlockDiag(LinearOperator):
    """Frontend class for ``LinearOperatorBlockDiag``."""


class LinearOperatorBlockLowerTriangular(LinearOperator):
    """Frontend class for ``LinearOperatorBlockLowerTriangular``."""


class LinearOperatorCirculant(LinearOperator):
    """Frontend class for ``LinearOperatorCirculant``."""


class LinearOperatorCirculant2D(LinearOperator):
    """Frontend class for ``LinearOperatorCirculant2D``."""


class LinearOperatorCirculant3D(LinearOperator):
    """Frontend class for ``LinearOperatorCirculant3D``."""


class LinearOperatorComposition(LinearOperator):
    """Frontend class for ``LinearOperatorComposition``."""


class LinearOperatorDiag(LinearOperator):
    """Frontend class for ``LinearOperatorDiag``."""


class LinearOperatorFullMatrix(LinearOperator):
    """Frontend class for ``LinearOperatorFullMatrix``."""


class LinearOperatorHouseholder(LinearOperator):
    """Frontend class for ``LinearOperatorHouseholder``."""


class LinearOperatorIdentity(LinearOperator):
    """Frontend class for ``LinearOperatorIdentity``."""


class LinearOperatorInversion(LinearOperator):
    """Frontend class for ``LinearOperatorInversion``."""


class LinearOperatorKronecker(LinearOperator):
    """Frontend class for ``LinearOperatorKronecker``."""


class LinearOperatorLowRankUpdate(LinearOperator):
    """Frontend class for ``LinearOperatorLowRankUpdate``."""


class LinearOperatorLowerTriangular(LinearOperator):
    """Frontend class for ``LinearOperatorLowerTriangular``."""


class LinearOperatorPermutation(LinearOperator):
    """Frontend class for ``LinearOperatorPermutation``."""


class LinearOperatorScaledIdentity(LinearOperator):
    """Frontend class for ``LinearOperatorScaledIdentity``."""


class LinearOperatorToeplitz(LinearOperator):
    """Frontend class for ``LinearOperatorToeplitz``."""


class LinearOperatorTridiag(LinearOperator):
    """Frontend class for ``LinearOperatorTridiag``."""


class LinearOperatorZeros(LinearOperator):
    """Frontend class for ``LinearOperatorZeros``."""
