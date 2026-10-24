"""Patches for ml-switcheroo-compiler missing eager registrations."""

from __future__ import annotations

import ml_switcheroo_compiler.backends.numpy.eager.linalg_advanced as _linalg_advanced
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerValue,
    as_float,
    as_int,
    as_opt_int,
    as_shape,
    numpy_eager_registry,
)
from ml_switcheroo_compiler.backends.numpy.eager import as_ndarray
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol
from ml_switcheroo_compiler.core.tensor import Tensor


def _seed_from_key(backend_module: BackendModuleProtocol, key_arg: EagerValue) -> int:
    """Derive a deterministic 32-bit unsigned integer seed from a PRNG key.

    Args:
        backend_module: Active NumPy backend module.
        key_arg: Input PRNG key value or array.

    Returns:
        int: Deterministic non-negative integer seed in ``[0, 2**32 - 1]``.
    """
    raw = key_arg.data if isinstance(key_arg, Tensor) else key_arg
    arr = backend_module.asarray(raw).reshape(-1)
    if arr.size == 0:
        return 0
    total = 0
    for idx in range(int(arr.size)):
        val = int(arr[idx])
        total = ((total * 1664525) + val + 1013904223 + idx) & 0xFFFFFFFF
    return total


def _np_lstsq(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate least-squares solution using ``numpy.linalg.lstsq``.

    Args:
        backend_module: Active NumPy module.
        *args: Positional arguments ``(a, b)``.
        **kwargs: Keyword arguments ``rcond``.

    Returns:
        EagerValue: Least-squares solution array.
    """
    sol = backend_module.linalg.lstsq(as_ndarray(args[0]), as_ndarray(args[1]), rcond=kwargs.get("rcond"))
    return as_ndarray(sol[0])


def _np_matrix_norm(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate matrix norm using ``numpy.linalg.norm``.

    Args:
        backend_module: Active NumPy module.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        EagerValue: Matrix norm result.
    """
    return as_ndarray(backend_module.linalg.norm(as_ndarray(args[0]), keepdims=bool(kwargs.get("keepdims", False))))


def _np_matrix_rank(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate matrix rank using ``numpy.linalg.matrix_rank``.

    Args:
        backend_module: Active NumPy module.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        EagerValue: Matrix rank result.
    """
    return as_ndarray(backend_module.linalg.matrix_rank(as_ndarray(args[0]), tol=kwargs.get("tol")))


def _np_vector_norm(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate vector norm using ``numpy.linalg.norm``.

    Args:
        backend_module: Active NumPy module.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        EagerValue: Vector norm result.
    """
    return as_ndarray(
        backend_module.linalg.norm(
            as_ndarray(args[0]),
            ord=kwargs.get("ord"),
            axis=as_opt_int(kwargs.get("axis")),
            keepdims=bool(kwargs.get("keepdims", False)),
        )
    )


def _np_random_normal(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate deterministic normal distribution sampling seeded by PRNG key.

    Args:
        backend_module: Active NumPy module.
        *args: Positional arguments where ``args[0]`` is the PRNG key.
        **kwargs: Keyword arguments ``shape`` and ``dtype``.

    Returns:
        EagerValue: Sampled normal array.
    """
    shape = as_shape(kwargs.get("shape", ()))
    seed = _seed_from_key(backend_module, args[0]) if args else 0
    rng = backend_module.random.RandomState(seed)
    return as_ndarray(rng.standard_normal(size=shape))


def _np_random_randint(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate deterministic uniform integer sampling seeded by PRNG key.

    Args:
        backend_module: Active NumPy module.
        *args: Positional arguments where ``args[0]`` is the PRNG key.
        **kwargs: Keyword arguments ``minval``, ``maxval``, and ``shape``.

    Returns:
        EagerValue: Sampled integer array.
    """
    low = as_int(kwargs.get("minval", 0))
    high = as_int(kwargs.get("maxval", 10))
    shape = as_shape(kwargs.get("shape", ()))
    seed = _seed_from_key(backend_module, args[0]) if args else 0
    rng = backend_module.random.RandomState(seed)
    return as_ndarray(rng.randint(low, high, size=shape))


def _np_random_bernoulli(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate deterministic Bernoulli distribution sampling seeded by PRNG key.

    Args:
        backend_module: Active NumPy module.
        *args: Positional arguments where ``args[0]`` is the PRNG key.
        **kwargs: Keyword arguments ``p`` and ``shape``.

    Returns:
        EagerValue: Sampled boolean array.
    """
    p = as_float(kwargs.get("p", 0.5))
    shape = as_shape(kwargs.get("shape", ()))
    seed = _seed_from_key(backend_module, args[0]) if args else 0
    rng = backend_module.random.RandomState(seed)
    return as_ndarray(rng.uniform(0.0, 1.0, size=shape) < p)


def register_compiler_patches() -> None:
    """Register all NumPy eager fallback handlers into ``numpy_eager_registry``."""
    numpy_eager_registry.register("Lstsq")(_np_lstsq)
    numpy_eager_registry.register("MatrixNorm")(_np_matrix_norm)
    numpy_eager_registry.register("MatrixRank")(_np_matrix_rank)
    numpy_eager_registry.register("VectorNorm")(_np_vector_norm)
    numpy_eager_registry.register("RandomNormal")(_np_random_normal)
    numpy_eager_registry.register("RandomRandint")(_np_random_randint)
    numpy_eager_registry.register("RandomBernoulli")(_np_random_bernoulli)


register_compiler_patches()

__all__ = [
    "_linalg_advanced",
    "_np_lstsq",
    "_np_matrix_norm",
    "_np_matrix_rank",
    "_np_random_bernoulli",
    "_np_random_normal",
    "_np_random_randint",
    "_np_vector_norm",
    "_seed_from_key",
    "numpy_eager_registry",
    "register_compiler_patches",
]
