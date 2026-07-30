"""Frontend API routing for jax.scipy.sparse.linalg."""

from typing import Any

import zero_jax._compiler_proxy_ops as _ops


def bicgstab(*args: Any, **kwargs: Any) -> Any:
    """Use Bi-Conjugate Gradient Stable iteration to solve ``Ax = b``."""
    return _ops.bicgstab(*args, **kwargs)


def cg(*args: Any, **kwargs: Any) -> Any:
    """Use Conjugate Gradient iteration to solve ``Ax = b``."""
    return _ops.cg(*args, **kwargs)


def gmres(*args: Any, **kwargs: Any) -> Any:
    """GMRES solves the linear system A x = b for x, given A and b."""
    return _ops.gmres(*args, **kwargs)


class LinearOperator:
    pass


class LinearOperatorAdjoint(LinearOperator):
    pass


class LinearOperatorBlockDiag(LinearOperator):
    pass


class LinearOperatorBlockLowerTriangular(LinearOperator):
    pass


class LinearOperatorCirculant(LinearOperator):
    pass


class LinearOperatorCirculant2D(LinearOperator):
    pass


class LinearOperatorCirculant3D(LinearOperator):
    pass


class LinearOperatorComposition(LinearOperator):
    pass


class LinearOperatorDiag(LinearOperator):
    pass


class LinearOperatorFullMatrix(LinearOperator):
    pass


class LinearOperatorHouseholder(LinearOperator):
    pass


class LinearOperatorIdentity(LinearOperator):
    pass


class LinearOperatorInversion(LinearOperator):
    pass


class LinearOperatorKronecker(LinearOperator):
    pass


class LinearOperatorLowRankUpdate(LinearOperator):
    pass


class LinearOperatorLowerTriangular(LinearOperator):
    pass


class LinearOperatorPermutation(LinearOperator):
    pass


class LinearOperatorScaledIdentity(LinearOperator):
    pass


class LinearOperatorToeplitz(LinearOperator):
    pass


class LinearOperatorTridiag(LinearOperator):
    pass


class LinearOperatorZeros(LinearOperator):
    pass
