"""Frontend API routing for jax.scipy.sparse.linalg."""

from typing import Any


def bicgstab(*args: Any, **kwargs: Any) -> Any:
    """Use Bi-Conjugate Gradient Stable iteration to solve ``Ax = b``."""
    raise NotImplementedError("bicgstab not yet implemented in zero-jax")


def cg(*args: Any, **kwargs: Any) -> Any:
    """Use Conjugate Gradient iteration to solve ``Ax = b``."""
    raise NotImplementedError("cg not yet implemented in zero-jax")


def gmres(*args: Any, **kwargs: Any) -> Any:
    """GMRES solves the linear system A x = b for x, given A and b."""
    raise NotImplementedError("gmres not yet implemented in zero-jax")
