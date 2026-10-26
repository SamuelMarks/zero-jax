"""Indexing, sorting, searching, and set operations for zero_jax.numpy."""

from __future__ import annotations

import builtins
from collections.abc import Callable, Sequence

import ml_switcheroo_compiler as _ml_switcheroo_compiler
from ml_switcheroo_compiler import Tensor
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import TensorConfig as _TensorConfig

import zero_jax._compiler_proxy_ops as ops
from zero_jax._types import (
    ArrayLikeSpec,
    ArrayProtocol,
    ConfigValue,
    DTypeSpec,
    VariadicCallable,
)
from zero_jax.numpy._creation import array, asarray, empty
from zero_jax.numpy._dtypes import int32
from zero_jax.numpy._manipulation import atleast_1d, column_stack, concatenate, ravel, reshape, swapaxes
from zero_jax.numpy._ndarray import _to_tensor, _wrap, ndarray
from zero_jax.numpy.constants import c_, index_exp, r_, s_
from zero_jax.numpy.tensor_utils import to_array as _to_array


def where(condition: ArrayLikeSpec, x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return elements chosen from x or y depending on condition.

    Args:
        condition (ArrayLikeSpec): Argument condition.
        x (ArrayLikeSpec): Argument x.
        y (ArrayLikeSpec): Argument y.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.where(_to_tensor(condition), _to_tensor(x), _to_tensor(y)))


def nonzero(a: ArrayLikeSpec, *, size: ArrayLikeSpec = None, fill_value: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``nonzero``.

    Args:
        a: Input parameter ``a``.
        size: Input parameter ``size``.
        fill_value: Input parameter ``fill_value``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    if size is None and fill_value is None:
        return _wrap(ops.nonzero(_to_tensor(a)))
    return _wrap(ops.nonzero(_to_tensor(a), size=size, fill_value=fill_value))


def flatnonzero(a: ArrayLikeSpec, *, size: ArrayLikeSpec = None, fill_value: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return indices that are non-zero in the flattened version of a.

    Args:
        a: Input array.
        size: Optional output size.
        fill_value: Optional fill value.

    Returns:
        Output array, containing the indices of the elements of a.ravel() that are non-zero.
    """
    nz = nonzero(ravel(a), size=size, fill_value=fill_value)
    return nz[0]


def argwhere(a: ArrayLikeSpec) -> ArrayLikeSpec:
    """Find the indices of array elements that are non-zero, grouped by element.

    Args:
        a: Input data.

    Returns:
        ArrayLikeSpec: Indices of non-zero elements with shape ``(N, a.ndim)``.
    """
    arr = asarray(a)
    if arr.ndim == 0:
        nz = nonzero(atleast_1d(arr))
        return empty((nz[0].shape[0], 0), dtype=int32)
    nz = nonzero(arr)
    return column_stack(nz) if len(nz) > 1 else reshape(nz[0], (-1, 1))


def searchsorted(a: ArrayLikeSpec, v: ArrayLikeSpec, side: str = "left", sorter: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Find indices where elements should be inserted to maintain order.

    Args:
        a: Input parameter ``a``.
        v: Input parameter ``v``.
        side: Input parameter ``side``.
        sorter: Input parameter ``sorter``.

    Returns:
        The operation result.
    """
    if sorter is not None:
        a = take(a, sorter, axis=-1)
    return _wrap(ops.searchsorted(_to_tensor(a), _to_tensor(v), side=side))


def take(a: ArrayLikeSpec, indices: ArrayLikeSpec, axis: int | None = None, mode: str | None = None) -> ArrayLikeSpec:
    """Take elements from an array along an axis.

    Args:
        a (ArrayLikeSpec): Argument a.
        indices (ArrayLikeSpec): Argument indices.
        axis (ArrayLikeSpec): Argument axis.
        mode (ArrayLikeSpec): Argument mode.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    _ = mode
    if axis is not None:
        return _wrap(ops.take(_to_tensor(a), _to_tensor(indices), axis=axis))
    return _wrap(ops.take(_to_tensor(a), _to_tensor(indices)))


def take_along_axis(arr: ArrayLikeSpec, indices: ArrayLikeSpec, axis: int) -> ArrayLikeSpec:
    """Take values from the input array by matching 1d index and data slices.

    Args:
        arr (ArrayLikeSpec): Argument arr.
        indices (ArrayLikeSpec): Argument indices.
        axis (ArrayLikeSpec): Argument axis.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.take_along_axis(_to_tensor(arr), _to_tensor(indices), axis=axis))


def put(
    a: ArrayLikeSpec,
    ind: ArrayLikeSpec,
    v: ArrayLikeSpec,
    mode: str = "raise",
    *,
    inplace: builtins.bool = False,
) -> ndarray:
    """Return a copy of ``a`` with target flat indices ``ind`` replaced by ``v``.

    Args:
        a: Target array.
        ind: Target flat indices.
        v: Values to place into ``a`` at ``ind``.
        mode: Out-of-bounds index handling mode (``"raise"``, ``"wrap"``, ``"clip"``).
        inplace: Must be ``False`` for immutable JAX arrays.

    Returns:
        ndarray: Functional copy of ``a`` with updated values.

    Raises:
        TypeError: If ``inplace=True`` is specified.
    """
    if inplace:
        raise TypeError("jax.numpy.put requires inplace=False because JAX arrays are immutable.")
    arr_obj = asarray(a)
    base = arr_obj.__array__().copy()
    flat = base.ravel()
    idx_np = asarray(ind).__array__().ravel()
    val_np = asarray(v).__array__().ravel()
    for i, raw_idx in enumerate(idx_np):
        idx_int = int(raw_idx)
        if mode == "wrap":
            idx_int = idx_int % flat.size
        elif mode == "clip":
            idx_int = max(0, min(idx_int, flat.size - 1))
        flat[idx_int] = val_np[i % val_np.size]
    return array(flat.reshape(base.shape), dtype=arr_obj.dtype)


def put_along_axis(
    arr: ArrayLikeSpec,
    indices: ArrayLikeSpec,
    values: ArrayLikeSpec,
    axis: int | None,
    inplace: builtins.bool = False,
    *,
    mode: str | None = None,
) -> ndarray:
    """Put ``values`` into the destination array ``arr`` by matching 1D index and data slices.

    Args:
        arr: Destination array.
        indices: Indices along each 1D slice of ``arr``.
        values: Values to insert at those indices.
        axis: Axis along which to index (or ``None`` for flattened indexing).
        inplace: Must be ``False`` for immutable JAX arrays.
        mode: Out-of-bounds index handling mode.

    Returns:
        ndarray: Updated copy of ``arr``.

    Raises:
        TypeError: If ``inplace=True`` is specified.
    """
    _ = mode
    if inplace:
        raise TypeError("jax.numpy.put_along_axis requires inplace=False because JAX arrays are immutable.")
    arr_obj = asarray(arr)
    base = arr_obj.__array__().copy()
    idx_np = asarray(indices).__array__()
    val_np = asarray(values).__array__()
    if axis is None:
        flat = base.ravel()
        flat[idx_np.ravel()] = val_np.ravel()
        return array(flat.reshape(base.shape), dtype=arr_obj.dtype)
    norm_ax = axis if axis >= 0 else arr_obj.ndim + axis
    moved_base = swapaxes(arr_obj, norm_ax, -1).__array__().copy()
    moved_idx = swapaxes(asarray(indices), norm_ax, -1).__array__()
    moved_val = swapaxes(asarray(values), norm_ax, -1).__array__()
    flat_base = moved_base.reshape(-1, moved_base.shape[-1])
    flat_idx = moved_idx.reshape(-1, moved_idx.shape[-1])
    flat_val = moved_val.reshape(-1, moved_val.shape[-1])
    rows = [[r] * flat_idx.shape[1] for r in builtins.range(flat_base.shape[0])]
    cols = [[int(flat_idx[r, c]) for c in builtins.range(flat_idx.shape[1])] for r in builtins.range(flat_base.shape[0])]
    vals = [[flat_val[r, c % flat_val.shape[1]] for c in builtins.range(flat_idx.shape[1])] for r in builtins.range(flat_base.shape[0])]
    flat_base[rows, cols] = vals
    restored = array(flat_base.reshape(moved_base.shape), dtype=arr_obj.dtype)
    return swapaxes(restored, norm_ax, -1)


def choose(a: ArrayLikeSpec, choices: ArrayLikeSpec, out: ArrayLikeSpec = None, mode: str = "raise") -> ArrayLikeSpec:
    """Construct an array from an index array and a set of arrays to choose from.

    Args:
        a: Array of integers indicating which array in choices to take values from.
        choices: Sequence of arrays to choose from.
        out: If provided, the result will be inserted into this array.
        mode: Specifies how indices outside [0, n-1] will be treated.

    Returns:
        The merged result.
    """
    c = [_to_tensor(ch) for ch in choices]
    return _wrap(ops.choose(_to_tensor(a), c, out=out, mode=mode))


def select(condlist: ArrayLikeSpec, choicelist: ArrayLikeSpec, default: ArrayLikeSpec = 0) -> ArrayLikeSpec:
    """JAX API implementation for select.

    Args:
        condlist: Argument condlist.
        choicelist: Argument choicelist.
        default: Argument default.

    Returns:
        ArrayLikeSpec: The result.
    """
    res = _to_tensor(default)
    for c, v in reversed(list(zip(condlist, choicelist))):
        res = ops.where(_to_tensor(c), _to_tensor(v), res)
    return _wrap(res)


def extract(condition: ArrayLikeSpec, arr: ArrayLikeSpec, *, size: ArrayLikeSpec = None, fill_value: ArrayLikeSpec = 0) -> ArrayLikeSpec:
    """Return the elements of an array that satisfy some condition.

    Args:
        condition: An array whose nonzero or True entries indicate the elements of arr to extract.
        arr: Input array of the same size as condition.
        size: Optional size limit.
        fill_value: Optional fill value.

    Returns:
        Rank 1 array of values from arr where condition is True.
    """
    return _wrap(ops.extract(_to_tensor(condition), _to_tensor(arr)))


def compress(
    condition: ArrayLikeSpec,
    a: ArrayLikeSpec,
    axis: ArrayLikeSpec = None,
    *,
    size: ArrayLikeSpec = None,
    fill_value: ArrayLikeSpec = 0,
    out: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Return selected slices of an array along given axis.

    Args:
        condition: Array that selects which entries to return.
        a: Input array.
        axis: Axis along which to take slices.
        size: Optional padding.
        fill_value: Padding value.
        out: Output array.

    Returns:
        A copy of a without the slices along axis for which condition is false.
    """
    return _wrap(ops.compress(_to_tensor(condition), _to_tensor(a), axis=axis, out=out))


def place(
    arr: ArrayLikeSpec,
    mask: ArrayLikeSpec,
    vals: ArrayLikeSpec,
    *,
    inplace: builtins.bool = False,
) -> ndarray:
    """Return a copy of ``arr`` with elements where ``mask`` is True replaced by ``vals``.

    Args:
        arr: Input array.
        mask: Boolean mask array with the same number of elements as ``arr``.
        vals: Values to put into ``arr`` where ``mask`` is True.
        inplace: Must be ``False`` for immutable JAX arrays.

    Returns:
        ndarray: Updated copy of ``arr``.

    Raises:
        TypeError: If ``inplace=True`` is specified.
    """
    if inplace:
        raise TypeError("jax.numpy.place requires inplace=False because JAX arrays are immutable.")
    arr_obj = asarray(arr)
    base = arr_obj.__array__().copy()
    flat = base.ravel()
    mask_np = asarray(mask).__array__().ravel()
    val_np = asarray(vals).__array__().ravel()
    v_idx = 0
    for i, m in enumerate(mask_np):
        if builtins.bool(m):
            flat[i] = val_np[v_idx % val_np.size]
            v_idx += 1
    return array(flat.reshape(base.shape), dtype=arr_obj.dtype)


def piecewise(
    x: ArrayLikeSpec,
    condlist: ArrayLikeSpec,
    funclist: ArrayLikeSpec,
    *args: ArrayLikeSpec,
    **kw: ArrayLikeSpec,
) -> ArrayLikeSpec:
    """Evaluate a piecewise-defined function.

    Args:
        x: The input domain.
        condlist: Each boolean array corresponds to a function in ``funclist``.
        funclist: Each function is evaluated over ``x`` wherever its corresponding condition is true.
        *args: Any further arguments given to ``piecewise`` are passed to the functions upon execution.
        **kw: Keyword arguments used in calling ``piecewise`` are passed to the functions upon execution.

    Returns:
        ArrayLikeSpec: The output array of the same shape and type as ``x``.
    """
    arr = asarray(x)
    conds = [condlist] if isinstance(condlist, ndarray) and condlist.ndim == arr.ndim else list(condlist)
    funcs = list(funclist)
    default_fn = funcs[-1] if len(funcs) == len(conds) + 1 else 0.0
    default_val = default_fn(arr, *args, **kw) if callable(default_fn) else default_fn
    choices = [f(arr, *args, **kw) if callable(f) else f for f in funcs[: len(conds)]]
    return select(conds, choices, default=default_val)


def diag_indices(n: int, ndim: int = 2) -> ArrayLikeSpec:
    """Return the indices to access the main diagonal of an array.

    Args:
        n: The size, along each dimension, of the arrays for which the returned indices can be used.
        ndim: The number of dimensions.

    Returns:
        A tuple of indices.
    """
    return tuple(_wrap(t) for t in ops.diag_indices(n, ndim=ndim))


def diag_indices_from(arr: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the indices to access the main diagonal of an n-dimensional array.

    Args:
        arr: Input array.

    Returns:
        A tuple of indices.
    """
    return tuple(_wrap(t) for t in ops.diag_indices_from(_to_tensor(arr)))


def tril_indices(n: int, k: int = 0, m: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``tril_indices``.

    Args:
        n: Input parameter ``n``.
        k: Input parameter ``k``.
        m: Input parameter ``m``.

    Returns:
        The operation result.
    """
    return _wrap(ops.tril_indices(n, k, m))


def tril_indices_from(arr: ArrayLikeSpec, k: int = 0) -> ArrayLikeSpec:
    """Execute ``tril_indices_from``.

    Args:
        arr: Input parameter ``arr``.
        k: Input parameter ``k``.

    Returns:
        The operation result.
    """
    return _wrap(ops.tril_indices_from(_to_tensor(arr), k))


def triu_indices(n: int, k: int = 0, m: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``triu_indices``.

    Args:
        n: Input parameter ``n``.
        k: Input parameter ``k``.
        m: Input parameter ``m``.

    Returns:
        The operation result.
    """
    return _wrap(ops.triu_indices(n, k, m))


def triu_indices_from(arr: ArrayLikeSpec, k: int = 0) -> ArrayLikeSpec:
    """Execute ``triu_indices_from``.

    Args:
        arr: Input parameter ``arr``.
        k: Input parameter ``k``.

    Returns:
        The operation result.
    """
    return _wrap(ops.triu_indices_from(_to_tensor(arr), k))


def mask_indices(
    n: int,
    mask_func: Callable[[ArrayLikeSpec, int], ArrayLikeSpec],
    k: int = 0,
) -> Sequence[ndarray]:
    """Return the indices to access ``(n, n)`` arrays, given a masking function.

    Args:
        n: The returned indices will be valid to access arrays of shape ``(n, n)``.
        mask_func: A function whose call signature is similar to that of ``triu``, ``tril``.
        k: An optional argument which is passed through to ``mask_func``.

    Returns:
        Sequence[ndarray]: Tuple of index arrays corresponding to non-zero mask locations.
    """
    return tuple(_wrap(t) for t in ops.mask_indices(n, mask_func, k))


def ravel_multi_index(multi_index: ArrayLikeSpec, dims: ArrayLikeSpec, mode: str = "raise", order: str = "C") -> ArrayLikeSpec:
    """JAX API implementation for ravel_multi_index.

    Args:
        multi_index: Input parameter ``multi_index``.
        dims: Input parameter ``dims``.
        mode: Input parameter ``mode``.
        order: Input parameter ``order``.

    Returns:
        The operation result.
    """
    return _wrap(ops.ravel_multi_index(multi_index, dims, mode, order))


def unravel_index(indices: ArrayLikeSpec, shape: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``unravel_index``.

    Args:
        indices: Input parameter ``indices``.
        shape: Input parameter ``shape``.

    Returns:
        The operation result.
    """
    from ml_switcheroo_compiler.ops.base import get_op

    res = get_op("UnravelIndex")()(_to_tensor(indices), shape)
    return _wrap(res)


def ix_(*args: ArrayLikeSpec) -> tuple[ndarray, ...]:
    """Construct an open mesh from multiple sequences.

    Args:
        *args: 1-D sequences.

    Returns:
        tuple[ndarray, ...]: Tuple of open meshgrid arrays.
    """
    from zero_jax.numpy._creation import meshgrid

    return tuple(meshgrid(*[asarray(a) for a in args], sparse=True, indexing="ij"))


def unique(
    ar: ArrayLikeSpec,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: ArrayLikeSpec = None,
    *,
    size: ArrayLikeSpec = None,
    fill_value: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Execute ``unique``.

    Args:
        ar: Input parameter ``ar``.
        return_index: Input parameter ``return_index``.
        return_inverse: Input parameter ``return_inverse``.
        return_counts: Input parameter ``return_counts``.
        axis: Input parameter ``axis``.
        size: Input parameter ``size``.
        fill_value: Input parameter ``fill_value``.

    Returns:
        The operation result.
    """
    _ = (size, fill_value)
    return _wrap(
        ops.unique(
            _to_tensor(ar),
            return_index,
            return_inverse,
            return_counts,
            axis,
        )
    )


def unique_all(x: ArrayLikeSpec, *, size: ArrayLikeSpec = None, fill_value: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``unique_all``.

    Args:
        x: Input parameter ``x``.
        size: Input parameter ``size``.
        fill_value: Input parameter ``fill_value``.

    Returns:
        The operation result.
    """
    return unique(x, return_index=True, return_inverse=True, return_counts=True)


def unique_counts(x: ArrayLikeSpec, *, size: ArrayLikeSpec = None, fill_value: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``unique_counts``.

    Args:
        x: Input parameter ``x``.
        size: Input parameter ``size``.
        fill_value: Input parameter ``fill_value``.

    Returns:
        The operation result.
    """
    return unique(x, return_counts=True)


def unique_inverse(x: ArrayLikeSpec, *, size: ArrayLikeSpec = None, fill_value: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``unique_inverse``.

    Args:
        x: Input parameter ``x``.
        size: Input parameter ``size``.
        fill_value: Input parameter ``fill_value``.

    Returns:
        The operation result.
    """
    return unique(x, return_inverse=True)


def unique_values(x: ArrayLikeSpec, *, size: ArrayLikeSpec = None, fill_value: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``unique_values``.

    Args:
        x: Input parameter ``x``.
        size: Input parameter ``size``.
        fill_value: Input parameter ``fill_value``.

    Returns:
        The operation result.
    """
    return unique(x)


def isin(element: ArrayLikeSpec, test_elements: ArrayLikeSpec, assume_unique: bool = False, invert: bool = False) -> ArrayLikeSpec:
    """Calculates element in test_elements, broadcasting over element only.

    Args:
        element: Input array.
        test_elements: The values against which to test each value of element.
        assume_unique: If True, the input arrays are both assumed to be unique.
        invert: If True, the values in the returned array are inverted.

    Returns:
        Has the same shape as element.
    """
    return _wrap(
        ops.isin(
            _to_tensor(element),
            _to_tensor(test_elements),
            assume_unique=assume_unique,
            invert=invert,
        )
    )


def intersect1d(ar1: ArrayLikeSpec, ar2: ArrayLikeSpec, assume_unique: bool = False, return_indices: bool = False) -> ArrayLikeSpec:
    """Find the intersection of two arrays.

    Args:
        ar1: Input array.
        ar2: Input array.
        assume_unique: If True, the input arrays are both assumed to be unique.
        return_indices: If True, the indices which correspond to the intersection of the two arrays are returned.

    Returns:
        Sorted 1D array of common and unique elements.
    """
    _ = return_indices
    u1 = ravel(asarray(ar1)) if assume_unique else unique(ar1)
    u2 = ravel(asarray(ar2)) if assume_unique else unique(ar2)
    mask = isin(u1, u2, assume_unique=True)
    return sort(compress(mask, u1))


def setdiff1d(ar1: ArrayLikeSpec, ar2: ArrayLikeSpec, assume_unique: bool = False) -> ArrayLikeSpec:
    """Find the set difference of two arrays.

    Args:
        ar1: Input array.
        ar2: Input comparison array.
        assume_unique: If True, the input arrays are both assumed to be unique.

    Returns:
        1D array of values in ``ar1`` that are not in ``ar2``.
    """
    u1 = ravel(asarray(ar1)) if assume_unique else unique(ar1)
    u2 = ravel(asarray(ar2)) if assume_unique else unique(ar2)
    mask = isin(u1, u2, assume_unique=True, invert=True)
    return compress(mask, u1)


def setxor1d(ar1: ArrayLikeSpec, ar2: ArrayLikeSpec, assume_unique: bool = False) -> ArrayLikeSpec:
    """Find the set exclusive-or of two arrays.

    Args:
        ar1: Input array.
        ar2: Input array.
        assume_unique: If True, the input arrays are both assumed to be unique.

    Returns:
        Sorted 1D array of unique values that are in only one of the input arrays.
    """
    d1 = setdiff1d(ar1, ar2, assume_unique=assume_unique)
    d2 = setdiff1d(ar2, ar1, assume_unique=assume_unique)
    return sort(concatenate([d1, d2], axis=0))


def union1d(ar1: ArrayLikeSpec, ar2: ArrayLikeSpec) -> ArrayLikeSpec:
    """Find the union of two arrays.

    Args:
        ar1: Input array.
        ar2: Input array.

    Returns:
        Unique, sorted union of the input arrays.
    """
    return unique(concatenate([ravel(asarray(ar1)), ravel(asarray(ar2))], axis=0))


def sort(a: ArrayLikeSpec, axis: int = -1, kind: ArrayLikeSpec = None, order: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for sort.

    Args:
        a: Argument a.
        axis: Argument axis.
        kind: Argument kind.
        order: Argument order.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.sort(_to_tensor(a), axis=axis))


def argsort(a: ArrayLikeSpec, axis: int = -1, kind: ArrayLikeSpec = None, order: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Returns the indices that would sort an array.

    Args:
        a: Input parameter ``a``.
        axis: Input parameter ``axis``.
        kind: Input parameter ``kind``.
        order: Input parameter ``order``.

    Returns:
        The operation result.
    """
    return _wrap(ops.argsort(_to_tensor(a), axis=axis))


def lexsort(keys: ArrayLikeSpec, axis: int = -1) -> ArrayLikeSpec:
    """Perform an indirect stable sort using a sequence of keys.

    Args:
        keys: The k different "columns" to be sorted.
        axis: Axis to be indirectly sorted.

    Returns:
        Array of indices that sort the keys along the specified axis.
    """
    return _wrap(
        ops.lexsort(
            [_to_tensor(k) for k in keys] if isinstance(keys, (list, tuple)) else _to_tensor(keys),
            axis=axis,
        )
    )


def partition(a: ArrayLikeSpec, kth: ArrayLikeSpec, axis: int = -1) -> ArrayLikeSpec:
    """Execute ``partition``.

    Args:
        a: Input parameter ``a``.
        kth: Input parameter ``kth``.
        axis: Input parameter ``axis``.

    Returns:
        The operation result.
    """
    from ml_switcheroo_compiler.ops.shape.frontend import partition as _partition

    return _wrap(_partition(_to_tensor(a), kth=kth, axis=axis))


def argpartition(a: ArrayLikeSpec, kth: ArrayLikeSpec, axis: int = -1, kind: str = "introselect", order: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Perform an indirect partition along the given axis via argsort.

    Args:
        a: Array to sort/partition.
        kth: Element index(es) to partition by.
        axis: Axis along which to sort.
        kind: Selection algorithm name.
        order: Field order comparison specification.

    Returns:
        ArrayLikeSpec: Array of indices that partition ``a`` along ``axis``.
    """
    return argsort(a, axis=axis)


def sort_complex(a: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for sort_complex.

    Args:
        a: Input parameter ``a``.

    Returns:
        The operation result.
    """
    return _wrap(ops.sort_complex(_to_tensor(a)))


def top_k(
    a: ArrayLikeSpec,
    k: int,
    /,
    *,
    axis: int = -1,
    mode: str = "largest",
    sorted: builtins.bool = True,
) -> tuple[ArrayLikeSpec, ArrayLikeSpec]:
    """Return the ``k`` largest (or smallest) elements and their indices along ``axis``.

    Args:
        a: Input array.
        k: Number of top elements to extract.
        axis: Axis along which to find the top-k elements.
        mode: ``"largest"`` or ``"smallest"``.
        sorted: Whether the returned elements are sorted in order.

    Returns:
        tuple[ArrayLikeSpec, ArrayLikeSpec]: Tuple ``(values, indices)``.
    """
    _ = sorted
    arr = asarray(a)
    descending = mode != "smallest"
    order_idx = arr.argsort(axis=axis, descending=descending)
    norm_ax = axis if axis >= 0 else arr.ndim + axis
    slc = tuple(builtins.slice(0, int(k)) if i == norm_ax else builtins.slice(None) for i in builtins.range(arr.ndim))
    top_idx = order_idx[slc]
    top_vals = take_along_axis(arr, top_idx, axis=norm_ax)
    return top_vals, top_idx


def packbits(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, bitorder: str = "big") -> ArrayLikeSpec:
    """Execute ``packbits``.

    Args:
        a: Input parameter ``a``.
        axis: Input parameter ``axis``.
        bitorder: Input parameter ``bitorder``.

    Returns:
        The operation result.
    """
    return _wrap(ops.packbits(_to_tensor(a), axis, bitorder))


def unpackbits(a: ArrayLikeSpec, axis: ArrayLikeSpec = None, count: ArrayLikeSpec = None, bitorder: str = "big") -> ArrayLikeSpec:
    """Execute ``unpackbits``.

    Args:
        a: Input parameter ``a``.
        axis: Input parameter ``axis``.
        count: Input parameter ``count``.
        bitorder: Input parameter ``bitorder``.

    Returns:
        The operation result.
    """
    return _wrap(ops.unpackbits(_to_tensor(a), axis, count, bitorder))


def in1d(ar1: ArrayLikeSpec, ar2: ArrayLikeSpec, assume_unique: builtins.bool = False, invert: builtins.bool = False) -> ArrayLikeSpec:
    """Test whether each element of a 1-D array is also present in a second array.

    Args:
        ar1: Input array.
        ar2: The values against which to test each value of ``ar1``.
        assume_unique: If ``True``, the input arrays are both assumed to be unique.
        invert: If ``True``, the values in the returned array are inverted.

    Returns:
        ArrayLikeSpec: Boolean array of the same length as ``ar1``.
    """
    return ravel(isin(ar1, ar2, assume_unique=assume_unique, invert=invert))


__all__ = [
    "argpartition",
    "argsort",
    "argwhere",
    "c_",
    "choose",
    "compress",
    "diag_indices",
    "diag_indices_from",
    "extract",
    "flatnonzero",
    "in1d",
    "index_exp",
    "intersect1d",
    "isin",
    "ix_",
    "lexsort",
    "mask_indices",
    "nonzero",
    "packbits",
    "partition",
    "piecewise",
    "place",
    "put",
    "put_along_axis",
    "r_",
    "ravel_multi_index",
    "s_",
    "searchsorted",
    "select",
    "setdiff1d",
    "setxor1d",
    "sort",
    "sort_complex",
    "take",
    "take_along_axis",
    "top_k",
    "tril_indices",
    "tril_indices_from",
    "triu_indices",
    "triu_indices_from",
    "union1d",
    "unique",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "unpackbits",
    "unravel_index",
    "where",
]
