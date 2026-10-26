"""Patches for ml-switcheroo-compiler missing eager registrations."""

from __future__ import annotations

import math
import random as _py_random

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

from zero_jax._types import SliceSpec


def _seed_from_key(backend_module: BackendModuleProtocol, key_arg: EagerValue) -> int:
    """Derive a deterministic 32-bit unsigned integer seed from a PRNG key.

    Args:
        backend_module: Active backend module.
        key_arg: Input PRNG key value or array.

    Returns:
        int: Deterministic non-negative integer seed in ``[0, 2**32 - 1]``.
    """
    arr = backend_module.asarray(key_arg).reshape(-1)
    if arr.size == 0:
        return 0
    total = 0
    for idx in range(int(arr.size)):
        val = int(arr[idx])
        total = ((total * 1664525) + val + 1013904223 + idx) & 0xFFFFFFFF
    return total


def _np_lstsq(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate least-squares solution using the active backend's ``linalg.lstsq``.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(a, b)``.
        **kwargs: Keyword arguments ``rcond``.

    Returns:
        EagerValue: Least-squares solution array.
    """
    sol = backend_module.linalg.lstsq(as_ndarray(args[0]), as_ndarray(args[1]), rcond=kwargs.get("rcond"))
    return as_ndarray(sol[0])


def _np_matrix_norm(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate matrix norm using the active backend's ``linalg.norm``.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        EagerValue: Matrix norm result.
    """
    return as_ndarray(backend_module.linalg.norm(as_ndarray(args[0]), keepdims=bool(kwargs.get("keepdims", False))))


def _np_matrix_rank(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate matrix rank using the active backend's ``linalg.matrix_rank``.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        EagerValue: Matrix rank result.
    """
    return as_ndarray(backend_module.linalg.matrix_rank(as_ndarray(args[0]), tol=kwargs.get("tol")))


def _np_vector_norm(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate vector norm using the active backend's ``linalg.norm``.

    Args:
        backend_module: Active backend module.
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
        backend_module: Active backend module.
        *args: Positional arguments where ``args[0]`` is the PRNG key.
        **kwargs: Keyword arguments ``shape`` and ``dtype``.

    Returns:
        EagerValue: Sampled normal array.
    """
    shape = as_shape(kwargs.get("shape", ()))
    seed = _seed_from_key(backend_module, args[0]) if args else 0
    rng = _py_random.Random(seed)
    count = math.prod(shape) if shape else 1
    samples = [rng.gauss(0.0, 1.0) for _ in range(count)]
    return as_ndarray(backend_module.asarray(samples).reshape(shape))


def _np_random_randint(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate deterministic uniform integer sampling seeded by PRNG key.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments where ``args[0]`` is the PRNG key.
        **kwargs: Keyword arguments ``minval``, ``maxval``, and ``shape``.

    Returns:
        EagerValue: Sampled integer array.
    """
    low = as_int(kwargs.get("minval", 0))
    high = as_int(kwargs.get("maxval", 10))
    shape = as_shape(kwargs.get("shape", ()))
    seed = _seed_from_key(backend_module, args[0]) if args else 0
    rng = _py_random.Random(seed)
    count = math.prod(shape) if shape else 1
    samples = [rng.randrange(low, high) for _ in range(count)]
    return as_ndarray(backend_module.asarray(samples).reshape(shape))


def _np_random_bernoulli(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate deterministic Bernoulli distribution sampling seeded by PRNG key.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments where ``args[0]`` is the PRNG key.
        **kwargs: Keyword arguments ``p`` and ``shape``.

    Returns:
        EagerValue: Sampled boolean array.
    """
    p = as_float(kwargs.get("p", 0.5))
    shape = as_shape(kwargs.get("shape", ()))
    seed = _seed_from_key(backend_module, args[0]) if args else 0
    rng = _py_random.Random(seed)
    count = math.prod(shape) if shape else 1
    samples = [rng.random() < p for _ in range(count)]
    return as_ndarray(backend_module.asarray(samples).reshape(shape))


def _np_lgamma(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate elementwise log-abs-gamma ``lgamma(x)``.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments where ``args[0]`` is ``x``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Log-abs-gamma evaluated elementwise.
    """
    arr = as_ndarray(args[0])
    flat_vals = [math.lgamma(float(val)) for val in arr.reshape(-1)]
    return as_ndarray(backend_module.asarray(flat_vals, dtype=arr.dtype).reshape(arr.shape))


def _np_dynamic_index_in_dim(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate ``dynamic_index_in_dim`` along a single axis.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(operand, index, dimension, keepdims)``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Indexed array or scalar.
    """
    _ = kwargs
    operand = as_ndarray(args[0])
    idx = int(backend_module.asarray(args[1]).reshape(-1)[0])
    dim = int(args[2]) if len(args) > 2 else 0
    keepdims = bool(args[3]) if len(args) > 3 else True
    axis = dim if dim >= 0 else operand.ndim + dim
    idx = max(0, min(idx if idx >= 0 else operand.shape[axis] + idx, operand.shape[axis] - 1))
    slc: list[SliceSpec | int] = [slice(None)] * operand.ndim
    slc[axis] = slice(idx, idx + 1) if keepdims else idx
    return as_ndarray(backend_module.asarray(operand[tuple(slc)]))


def _np_dynamic_slice_in_dim(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate ``dynamic_slice_in_dim`` along a single axis.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(operand, start_index, slice_size, dimension)``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Sliced array.
    """
    _ = kwargs
    operand = as_ndarray(args[0])
    start = int(backend_module.asarray(args[1]).reshape(-1)[0])
    size = int(args[2])
    dim = int(args[3]) if len(args) > 3 else 0
    axis = dim if dim >= 0 else operand.ndim + dim
    start = max(0, min(start if start >= 0 else operand.shape[axis] + start, max(0, operand.shape[axis] - size)))
    slc: list[SliceSpec] = [slice(None)] * operand.ndim
    slc[axis] = slice(start, start + size)
    return as_ndarray(operand[tuple(slc)])


def _np_dynamic_update_index_in_dim(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate ``dynamic_update_index_in_dim`` along a single axis.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(operand, update, index, dimension)``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Updated copy of ``operand``.
    """
    _ = kwargs
    operand = as_ndarray(args[0]).copy()
    update = as_ndarray(args[1])
    idx = int(backend_module.asarray(args[2]).reshape(-1)[0])
    dim = int(args[3]) if len(args) > 3 else 0
    axis = dim if dim >= 0 else operand.ndim + dim
    idx = max(0, min(idx if idx >= 0 else operand.shape[axis] + idx, operand.shape[axis] - 1))
    slc: list[SliceSpec | int] = [slice(None)] * operand.ndim
    slc[axis] = idx
    operand[tuple(slc)] = update
    return as_ndarray(operand)


def _np_dynamic_update_slice_in_dim(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate ``dynamic_update_slice_in_dim`` along a single axis.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(operand, update, start_index, dimension)``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Updated copy of ``operand``.
    """
    _ = kwargs
    operand = as_ndarray(args[0]).copy()
    update = as_ndarray(args[1])
    start = int(backend_module.asarray(args[2]).reshape(-1)[0])
    dim = int(args[3]) if len(args) > 3 else 0
    axis = dim if dim >= 0 else operand.ndim + dim
    size = update.shape[axis]
    start = max(0, min(start if start >= 0 else operand.shape[axis] + start, max(0, operand.shape[axis] - size)))
    slc: list[SliceSpec] = [slice(None)] * operand.ndim
    slc[axis] = slice(start, start + size)
    operand[tuple(slc)] = update
    return as_ndarray(operand)


def _np_scatter_add(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate ``scatter_add`` by accumulating ``updates`` into ``operand`` at ``indices``.

    Args:
        backend_module: Active backend module.
        *args: Positional arguments ``(operand, axis, indices, updates)``.
        **kwargs: Unused keyword arguments.

    Returns:
        EagerValue: Updated array with accumulated scatter values.
    """
    _ = kwargs
    operand = as_ndarray(args[0]).copy()
    indices = backend_module.asarray(args[2]).reshape(-1)
    updates = backend_module.asarray(args[3]).reshape(-1)
    flat_op = operand.reshape(-1)
    for i, idx_val in enumerate(indices):
        flat_op[int(idx_val) % flat_op.size] += updates[i % updates.size]
    return as_ndarray(operand)


def register_compiler_patches() -> None:
    """Register all eager fallback handlers into ``numpy_eager_registry``."""
    numpy_eager_registry.register("Lgamma")(_np_lgamma)
    numpy_eager_registry.register("Lstsq")(_np_lstsq)
    numpy_eager_registry.register("MatrixNorm")(_np_matrix_norm)
    numpy_eager_registry.register("MatrixRank")(_np_matrix_rank)
    numpy_eager_registry.register("VectorNorm")(_np_vector_norm)
    numpy_eager_registry.register("RandomNormal")(_np_random_normal)
    numpy_eager_registry.register("RandomRandint")(_np_random_randint)
    numpy_eager_registry.register("RandomBernoulli")(_np_random_bernoulli)
    numpy_eager_registry.register("DynamicIndexInDim")(_np_dynamic_index_in_dim)
    numpy_eager_registry.register("DynamicSliceInDim")(_np_dynamic_slice_in_dim)
    numpy_eager_registry.register("DynamicUpdateIndexInDim")(_np_dynamic_update_index_in_dim)
    numpy_eager_registry.register("DynamicUpdateSliceInDim")(_np_dynamic_update_slice_in_dim)
    numpy_eager_registry.register("ScatterAdd")(_np_scatter_add)


register_compiler_patches()

__all__ = [
    "_linalg_advanced",
    "_np_dynamic_index_in_dim",
    "_np_dynamic_slice_in_dim",
    "_np_dynamic_update_index_in_dim",
    "_np_dynamic_update_slice_in_dim",
    "_np_lgamma",
    "_np_lstsq",
    "_np_matrix_norm",
    "_np_matrix_rank",
    "_np_random_bernoulli",
    "_np_random_normal",
    "_np_random_randint",
    "_np_scatter_add",
    "_np_vector_norm",
    "_seed_from_key",
    "numpy_eager_registry",
    "register_compiler_patches",
]
