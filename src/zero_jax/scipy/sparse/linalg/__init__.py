"""Frontend API routing for jax.scipy.sparse.linalg."""

from typing import Any
import ml_switcheroo_compiler.ops as _ops


def bicgstab(*args: Any, **kwargs: Any) -> Any:
    """Use Bi-Conjugate Gradient Stable iteration to solve ``Ax = b``."""
    return getattr(_ops, "bicgstab")(*args, **kwargs)


def cg(*args: Any, **kwargs: Any) -> Any:
    """Use Conjugate Gradient iteration to solve ``Ax = b``."""
    return getattr(_ops, "cg")(*args, **kwargs)


def gmres(*args: Any, **kwargs: Any) -> Any:
    """GMRES solves the linear system A x = b for x, given A and b."""
    return getattr(_ops, "gmres")(*args, **kwargs)
