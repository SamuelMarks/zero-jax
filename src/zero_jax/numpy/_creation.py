"""Array creation routines for zero_jax.numpy."""

from __future__ import annotations

import builtins
from collections.abc import Callable

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
from zero_jax.numpy._dtypes import astype, float32, int32
from zero_jax.numpy._ndarray import _to_tensor, _wrap, ndarray
from zero_jax.numpy.tensor_utils import to_array as _to_array


def array(x: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Create an array.

    Args:
        x (ArrayLikeSpec): Argument x.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    res = x if isinstance(x, ndarray) else _wrap(_to_tensor(x))
    if dtype is not None:
        return astype(res, dtype)
    return res


def asarray(x: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Convert the input to an array.

    Args:
        x (ArrayLikeSpec): Argument x.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return array(x, dtype=dtype)


def zeros(shape: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return a new array of given shape and type, filled with zeros.

    Args:
        shape (ArrayLikeSpec): Argument shape.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(ops.zeros(shape=shape, dtype=dtype))


def ones(shape: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return a new array of given shape and type, filled with ones.

    Args:
        shape (ArrayLikeSpec): Argument shape.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(ops.ones(shape=shape, dtype=dtype))


def empty(shape: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return a new array of given shape and type, without initializing entries.

    Args:
        shape (ArrayLikeSpec): Argument shape.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(ops.empty(shape=shape, dtype=dtype))


def full(shape: ArrayLikeSpec, fill_value: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return a new array of given shape and type, filled with fill_value.

    Args:
        shape (ArrayLikeSpec): Argument shape.
        fill_value (ArrayLikeSpec): Argument fill_value.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.full(shape=shape, fill_value=fill_value, dtype=dtype))


def zeros_like(x: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return an array of zeros with the same shape and type as a given array.

    Args:
        x (ArrayLikeSpec): Argument x.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(ops.zeros_like(_to_tensor(x), dtype=dtype))


def ones_like(x: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return an array of ones with the same shape and type as a given array.

    Args:
        x (ArrayLikeSpec): Argument x.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(ops.ones_like(_to_tensor(x), dtype=dtype))


def empty_like(x: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return a new array with the same shape and type as a given array.

    Args:
        x (ArrayLikeSpec): Argument x.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    t = _to_tensor(x)
    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(ops.empty(shape=t.shape, dtype=dtype))


def full_like(x: ArrayLikeSpec, fill_value: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return a full array with the same shape and type as a given array.

    Args:
        x (ArrayLikeSpec): Argument x.
        fill_value (ArrayLikeSpec): Argument fill_value.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.full_like(_to_tensor(x), fill_value=fill_value, dtype=dtype))


def arange(start: ArrayLikeSpec, stop: ArrayLikeSpec = None, step: ArrayLikeSpec = 1, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return evenly spaced values within a given interval.

    Args:
        start (ArrayLikeSpec): Argument start.
        stop (ArrayLikeSpec): Argument stop.
        step (ArrayLikeSpec): Argument step.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    from zero_jax.nn.activation import _to_dtype

    resolved_dtype = _to_dtype(dtype) if dtype is not None else None
    return _wrap(ops.arange(start=start, stop=stop, step=step, dtype=resolved_dtype))


def linspace(
    start: ArrayLikeSpec,
    stop: ArrayLikeSpec,
    num: int = 50,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: ArrayLikeSpec = None,
    axis: int = 0,
) -> ArrayLikeSpec:
    """Return evenly spaced numbers over a specified interval.

    Args:
        start: Input parameter ``start``.
        stop: Input parameter ``stop``.
        num: Input parameter ``num``.
        endpoint: Input parameter ``endpoint``.
        retstep: Input parameter ``retstep``.
        dtype: Input parameter ``dtype``.
        axis: Input parameter ``axis``.

    Returns:
        The operation result.
    """
    if num < 0:
        raise ValueError(f"Number of samples, {num}, must be non-negative.")
    if num == 0:
        return _wrap(ops.empty((0,), dtype=dtype))

    _to_tensor(start)
    _to_tensor(stop)

    if dtype is None:
        import ml_switcheroo_compiler

        dtype = ml_switcheroo_compiler.core.dtype.DType.Float32

    step_denom = builtins.max(num - 1, 1) if endpoint else num

    step = (stop - start) / step_denom

    from ml_switcheroo_compiler.ops import (
        add as backend_add,
    )
    from ml_switcheroo_compiler.ops import (
        arange as backend_arange,
    )
    from ml_switcheroo_compiler.ops import (
        multiply as backend_mul,
    )

    idx = backend_arange(0, num, 1, dtype=dtype)
    res = backend_add(_to_tensor(start), backend_mul(idx, _to_tensor(step)))

    res_wrapped = _wrap(res)

    if axis != 0:
        res_wrapped = moveaxis(res_wrapped, 0, axis)

    if retstep:
        return res_wrapped, step
    return res_wrapped


def logspace(
    start: ArrayLikeSpec,
    stop: ArrayLikeSpec,
    num: int = 50,
    endpoint: bool = True,
    base: float = 10.0,
    dtype: ArrayLikeSpec = None,
    axis: int = 0,
) -> ArrayLikeSpec:
    """Return numbers spaced evenly on a log scale.

    Args:
        start (ArrayLikeSpec): Argument start.
        stop (ArrayLikeSpec): Argument stop.
        num (ArrayLikeSpec): Argument num.
        endpoint (ArrayLikeSpec): Argument endpoint.
        base (ArrayLikeSpec): Argument base.
        dtype (ArrayLikeSpec): Argument dtype.
        axis (ArrayLikeSpec): Argument axis.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    lin = linspace(start, stop, num, endpoint=endpoint, dtype=dtype, axis=axis)
    return power(base, lin)


def geomspace(
    start: ArrayLikeSpec,
    stop: ArrayLikeSpec,
    num: int = 50,
    endpoint: bool = True,
    dtype: ArrayLikeSpec = None,
    axis: int = 0,
) -> ArrayLikeSpec:
    """Return numbers spaced evenly on a log scale (a geometric progression).

    Args:
        start: The starting value of the sequence.
        stop: The final value of the sequence, unless ``endpoint=False``.
        num: Number of samples to generate.
        endpoint: If ``True``, ``stop`` is the last sample.
        dtype: The type of the output array.
        axis: The axis in the result to store the samples.

    Returns:
        ArrayLikeSpec: Geometrically spaced samples.
    """
    log_start = log(asarray(start, dtype=float32))
    log_stop = log(asarray(stop, dtype=float32))
    res = exp(linspace(log_start, log_stop, num=num, endpoint=endpoint, axis=axis))
    if dtype is not None:
        return astype(res, dtype)
    return res


def eye(N: int, M: int | None = None, k: int = 0, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Args:
        N: Input parameter ``N``.
        M: Input parameter ``M``.
        k: Input parameter ``k``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    if M is None:
        M = N
    if k == 0:
        return _wrap(ops.eye(N, M, dtype=dtype))

    row_idx = arange(N)
    col_idx = arange(M)
    row_k = add(row_idx, k)
    row_k_expand = expand_dims(row_k, 1)
    col_expand = expand_dims(col_idx, 0)
    res = equal(row_k_expand, col_expand)

    if dtype is None:
        import ml_switcheroo_compiler

        dtype = ml_switcheroo_compiler.core.dtype.DType.Float32
    return astype(res, dtype)


def identity(n: int, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Return the identity array.

    Args:
        n (ArrayLikeSpec): Argument n.
        dtype (ArrayLikeSpec): Argument dtype.

    Returns:
        ArrayLikeSpec: The result of the operation.
    """
    return _wrap(ops.identity(n=n, dtype=dtype))


def diag(v: ArrayLikeSpec, k: int = 0) -> ArrayLikeSpec:
    """JAX API implementation for diag.

    Args:
        v: Argument v.
        k: Argument k.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.diag(_to_tensor(v), diagonal=k))


def diagflat(v: ArrayLikeSpec, k: int = 0) -> ArrayLikeSpec:
    """Create a two-dimensional array with the flattened input as a diagonal.

    Args:
        v: Input data.
        k: Diagonal to set.

    Returns:
        The 2-D output array.
    """
    return _wrap(ops.diagflat(_to_tensor(v), k=k))


def tri(N: int, M: ArrayLikeSpec = None, k: int = 0, dtype: ArrayLikeSpec = float) -> ArrayLikeSpec:
    """Construct an array with ones at and below the given diagonal and zeros elsewhere.

    Args:
        N: Number of rows in the array.
        M: Number of columns in the array (defaults to ``N``).
        k: The sub-diagonal at and below which the array is filled.
        dtype: Data type of the returned array.

    Returns:
        ArrayLikeSpec: Lower-triangular indicator matrix of shape ``(N, M)``.
    """
    m_cols = int(N) if M is None else int(M)
    dt = float32 if dtype is float or dtype is None else dtype
    rows = arange(0, int(N), dtype=int32)[:, None]
    cols = arange(0, m_cols, dtype=int32)[None, :]
    return asarray(rows >= (cols - int(k)), dtype=dt)


def tril(m: ArrayLikeSpec, k: int = 0) -> ArrayLikeSpec:
    """JAX API implementation for tril.

    Args:
        m: Argument m.
        k: Argument k.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.tril(_to_tensor(m), diagonal=k))


def triu(m: ArrayLikeSpec, k: int = 0) -> ArrayLikeSpec:
    """JAX API implementation for triu.

    Args:
        m: Argument m.
        k: Argument k.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.triu(_to_tensor(m), diagonal=k))


def vander(x: ArrayLikeSpec, N: ArrayLikeSpec = None, increasing: bool = False) -> ArrayLikeSpec:
    """Generate a Vandermonde matrix.

    Args:
        x: 1-D input array.
        N: Number of columns in the output (defaults to ``len(x)``).
        increasing: Order of the powers of the columns.

    Returns:
        ArrayLikeSpec: Vandermonde matrix.
    """
    arr = asarray(x)
    n_cols = int(arr.shape[0]) if N is None else int(N)
    exps = arange(0, n_cols, dtype=arr.dtype) if increasing else arange(n_cols - 1, -1, -1, dtype=arr.dtype)
    return arr[:, None] ** exps[None, :]


def meshgrid(*xi: ArrayLikeSpec, copy: ArrayLikeSpec = True, sparse: ArrayLikeSpec = False, indexing: ArrayLikeSpec = "xy") -> ArrayLikeSpec:
    """Return coordinate matrices from coordinate vectors.

    Args:
        copy: Input parameter ``copy``.
        sparse: Input parameter ``sparse``.
        indexing: Input parameter ``indexing``.
        *xi: Input parameter ``xi``.

    Returns:
        The operation result.
    """
    tensors = [_to_tensor(x) for x in xi]

    ndim = len(tensors)
    if indexing == "xy" and ndim > 1:
        tensors[0], tensors[1] = tensors[1], tensors[0]

    s0 = (1,) * ndim
    output = []
    for i, t in enumerate(tensors):
        shape = list(s0)
        shape[i] = -1
        reshaped = ops.reshape(t, newshape=tuple(shape))
        output.append(reshaped)

    if not sparse:
        broadcast_shape = tuple(t.shape[0] for t in tensors)
        output = [ops.broadcast_to(t, broadcast_shape) for t in output]

    if indexing == "xy" and ndim > 1:
        output[0], output[1] = output[1], output[0]

    return tuple(_wrap(t) for t in output)


def indices(dimensions: ArrayLikeSpec, dtype: ArrayLikeSpec = None, sparse: bool = False) -> ArrayLikeSpec:
    """Return an array representing the indices of a grid.

    Args:
        dimensions: The shape of the grid.
        dtype: Data type of the result.
        sparse: Return a sparse representation of the grid instead of a dense representation.

    Returns:
        If sparse is False: one array of grid indices. If True: a tuple of arrays.
    """
    dims = tuple(int(d) for d in dimensions)
    dt = int32 if dtype is None else dtype
    axes = [arange(0, d, dtype=dt) for d in dims]
    grids = meshgrid(*axes, sparse=sparse, indexing="ij")
    if sparse:
        return grids
    return stack(list(grids), axis=0) if grids else empty((0,), dtype=dt)


def frombuffer(buffer: ArrayLikeSpec, dtype: ArrayLikeSpec = float, count: int = -1, offset: int = 0) -> ArrayLikeSpec:
    """Interpret a buffer as a 1-dimensional array.

    Args:
        buffer: Buffer to read.
        dtype: Data-type of the returned array.
        count: Number of items to read.
        offset: Start reading the buffer from this offset.

    Returns:
        The constructed array.
    """
    if isinstance(buffer, (bytes, bytearray, memoryview)):
        raw_bytes = bytes(buffer)[offset:]
        vals = list(raw_bytes)
        if count >= 0:
            vals = vals[:count]
        return array(vals, dtype=dtype)
    return array(buffer, dtype=dtype)


def fromfile(
    file: ArrayLikeSpec,
    dtype: ArrayLikeSpec = float,
    count: int = -1,
    sep: str = "",
    offset: int = 0,
    *,
    like: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Construct an array from data in a text or binary file.

    Args:
        file: Open file object or filename.
        dtype: Data type of the returned array.
        count: Number of items to read (``-1`` means all).
        sep: Separator between items if file is a text file.
        offset: Byte offset from the current file position.
        like: Reference object for array creation.

    Returns:
        ArrayLikeSpec: Array read from the file.
    """
    import os

    if isinstance(file, (str, bytes)) and os.path.exists(file):
        with open(file, "r" if sep else "rb") as f:
            data = f.read()
        if isinstance(data, str):
            return fromstring(data, dtype=dtype, count=count, sep=sep or " ")
        return frombuffer(data, dtype=dtype, count=count, offset=offset)
    raise FileNotFoundError(f"File not found: {file!r}")


def fromfunction(function: ArrayLikeSpec, shape: ArrayLikeSpec, *, dtype: ArrayLikeSpec = float, like: ArrayLikeSpec = None, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Construct an array by executing a function over each coordinate.

    Args:
        function: Callable accepting ``N`` coordinate arrays where ``N = len(shape)``.
        shape: Shape of the output array.
        dtype: Data-type of the coordinate arrays passed to ``function``.
        like: Reference object for array creation.
        **kwargs: Additional keyword arguments passed to ``function``.

    Returns:
        ArrayLikeSpec: Result of calling ``function`` on the coordinate grids.
    """
    _ = like
    coords = indices(shape, dtype=dtype)
    return asarray(function(*coords, **kwargs))


def fromiter(iter: ArrayLikeSpec, dtype: ArrayLikeSpec, count: int = -1, *, like: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Create a new 1-dimensional array from an iterable object.

    Args:
        iter: An iterable object providing data for the array.
        dtype: The data-type of the returned array.
        count: The number of items to read from ``iter`` (default ``-1`` reads all).
        like: Reference object for array creation.

    Returns:
        ArrayLikeSpec: 1-D array constructed from ``iter``.
    """
    items = list(iter)
    if count >= 0:
        items = items[:count]
    return array(items, dtype=dtype)


def fromstring(string: ArrayLikeSpec, dtype: ArrayLikeSpec = float, count: int = -1, sep: str = "") -> ArrayLikeSpec:
    """Initialize a new 1-D array from text data in a string.

    Args:
        string: A string containing the data.
        dtype: The data type of the array.
        count: Read this number of ``dtype`` elements from the data.
        sep: The string separating numbers in the data.

    Returns:
        ArrayLikeSpec: 1-D array constructed from ``string``.
    """
    if isinstance(string, str):
        parts = string.split(sep) if sep else string.split()
        if count >= 0:
            parts = parts[:count]
        vals = [float(p) for p in parts]
        return array(vals, dtype=dtype)
    return frombuffer(string, dtype=dtype, count=count)


def from_dlpack(x: ArrayLikeSpec, /, *, device: ArrayLikeSpec = None, copy: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Create an array from a DLPack tensor.

    Args:
        x: A DLPack tensor.
        device: The device on which the created array should be placed.
        copy: Optional.

    Returns:
        The array.
    """
    _ = (device, copy)
    return asarray(x)


def copy(a: ArrayLikeSpec, order: str = "K") -> ArrayLikeSpec:
    """Return an array copy of the given object.

    Args:
        a: Input data.
        order: memory layout.

    Returns:
        An array interpretation of a.
    """
    return _wrap(ops.array(_to_tensor(a)))


def diagonal(a: ArrayLikeSpec, offset: int = 0, axis1: int = 0, axis2: int = 1) -> ArrayLikeSpec:
    """Return specified diagonals.

    Args:
        a: Input array.
        offset: Offset of the diagonal from the main diagonal.
        axis1: Axis to be used as the first axis of the 2-D sub-arrays.
        axis2: Axis to be used as the second axis of the 2-D sub-arrays.

    Returns:
        Array of diagonals.
    """
    return _wrap(ops.diagonal(_to_tensor(a), offset=offset, axis1=axis1, axis2=axis2))


def fill_diagonal(a: ArrayLikeSpec, val: ArrayLikeSpec, wrap: bool = False, *, inplace: bool = True) -> ArrayLikeSpec:
    """Fill the main diagonal of the given array of any dimensionality.

    Args:
        a: Array whose diagonal is to be filled.
        val: Value to be written on the diagonal.
        wrap: For tall matrices in NumPy version up to 1.6.2, the diagonal "wrapped" after N columns.
        inplace: Modified in place.

    Returns:
        The array with the diagonal filled.
    """
    return _wrap(ops.fill_diagonal(_to_tensor(a), _to_tensor(val), wrap=wrap))


def frompyfunc(func: ArrayLikeSpec, nin: int, nout: int, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Take an arbitrary Python function and return a vectorized callable.

    Args:
        func: A Python function object.
        nin: The number of input arguments.
        nout: The number of objects returned by ``func``.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        ArrayLikeSpec: Vectorized wrapper around ``func``.
    """
    return vectorize(func)


class vectorize:
    """Generalized function class that vectorizes a Python callable ``pyfunc``."""

    def __init__(
        self,
        pyfunc: ArrayLikeSpec,
        otypes: ArrayLikeSpec = None,
        doc: ArrayLikeSpec = None,
        excluded: ArrayLikeSpec = None,
        cache: bool = False,
        signature: ArrayLikeSpec = None,
    ) -> None:
        """Initialize ``vectorize``.

        Args:
            pyfunc: Python function or method to vectorize.
            otypes: Optional output data types.
            doc: Optional docstring for the wrapper.
            excluded: Set of arguments to exclude from vectorization.
            cache: Whether to cache the first function evaluation.
            signature: Generalized universal function signature (e.g. ``"(m,n),(n)->(m)"``).
        """
        self.pyfunc = pyfunc
        self.otypes = otypes
        self.__doc__ = doc
        self.excluded = set(excluded) if isinstance(excluded, (set, list, tuple)) else set()
        self.cache = cache
        self.signature = signature

    def __call__(self, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
        """Invoke the vectorized callable on ``args`` and ``kwargs``.

        Args:
            *args: Positional array or scalar arguments.
            **kwargs: Keyword arguments.

        Returns:
            ArrayLikeSpec: Vectorized evaluation result.
        """
        if callable(self.pyfunc):
            res = self.pyfunc(*[asarray(a) if idx not in self.excluded else a for idx, a in enumerate(args)], **kwargs)
            return asarray(res) if not isinstance(res, tuple) else tuple(asarray(r) for r in res)
        raise TypeError("pyfunc must be callable")


from zero_jax.numpy._manipulation import expand_dims, moveaxis, stack
from zero_jax.numpy._math import add, equal, exp, log, power

__all__ = [
    "arange",
    "array",
    "asarray",
    "copy",
    "diag",
    "diagflat",
    "diagonal",
    "empty",
    "empty_like",
    "eye",
    "fill_diagonal",
    "from_dlpack",
    "frombuffer",
    "fromfile",
    "fromfunction",
    "fromiter",
    "frompyfunc",
    "fromstring",
    "full",
    "full_like",
    "geomspace",
    "identity",
    "indices",
    "linspace",
    "logspace",
    "meshgrid",
    "ones",
    "ones_like",
    "tri",
    "tril",
    "triu",
    "vander",
    "vectorize",
    "zeros",
    "zeros_like",
]
