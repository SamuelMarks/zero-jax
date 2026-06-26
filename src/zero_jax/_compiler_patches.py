"""Patches for ml-switcheroo-compiler missing eager registrations."""

from typing import Any
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Asin")
def _asin(np: Any, *args: Any, **kwargs: Any) -> Any:
    """Implementation of _asin.
    Args:
        np: The numpy module.
        *args: positional args.
        **kwargs: keyword args.
    Returns:
        The result.
    """
    return np.arcsin(*args, **kwargs)


@numpy_eager_registry.register("Acos")
def _acos(np: Any, *args: Any, **kwargs: Any) -> Any:
    """Implementation of _acos.
    Args:
        np: The numpy module.
        *args: positional args.
        **kwargs: keyword args.
    Returns:
        The result.
    """
    return np.arccos(*args, **kwargs)


@numpy_eager_registry.register("Atan")
def _atan(np: Any, *args: Any, **kwargs: Any) -> Any:
    """Implementation of _atan.
    Args:
        np: The numpy module.
        *args: positional args.
        **kwargs: keyword args.
    Returns:
        The result.
    """
    return np.arctan(*args, **kwargs)


@numpy_eager_registry.register("Atan2")
def _atan2(np: Any, *args: Any, **kwargs: Any) -> Any:
    """Implementation of _atan2.
    Args:
        np: The numpy module.
        *args: positional args.
        **kwargs: keyword args.
    Returns:
        The result.
    """
    return np.arctan2(*args, **kwargs)


@numpy_eager_registry.register("Asinh")
def _asinh(np: Any, *args: Any, **kwargs: Any) -> Any:
    """Implementation of _asinh.
    Args:
        np: The numpy module.
        *args: positional args.
        **kwargs: keyword args.
    Returns:
        The result.
    """
    return np.arcsinh(*args, **kwargs)


@numpy_eager_registry.register("Acosh")
def _acosh(np: Any, *args: Any, **kwargs: Any) -> Any:
    """Implementation of _acosh.
    Args:
        np: The numpy module.
        *args: positional args.
        **kwargs: keyword args.
    Returns:
        The result.
    """
    return np.arccosh(*args, **kwargs)


@numpy_eager_registry.register("Atanh")
def _atanh(np: Any, *args: Any, **kwargs: Any) -> Any:
    """Implementation of _atanh.
    Args:
        np: The numpy module.
        *args: positional args.
        **kwargs: keyword args.
    Returns:
        The result.
    """
    return np.arctanh(*args, **kwargs)


@numpy_eager_registry.register("Deg2Rad")
def _deg2rad(np: Any, *args: Any, **kwargs: Any) -> Any:
    """Implementation of _deg2rad.
    Args:
        np: The numpy module.
        *args: positional args.
        **kwargs: keyword args.
    Returns:
        The result.
    """
    return np.deg2rad(*args, **kwargs)


@numpy_eager_registry.register("Rad2Deg")
def _rad2deg(np: Any, *args: Any, **kwargs: Any) -> Any:
    """Implementation of _rad2deg.
    Args:
        np: The numpy module.
        *args: positional args.
        **kwargs: keyword args.
    Returns:
        The result.
    """
    return np.rad2deg(*args, **kwargs)


@numpy_eager_registry.register("Cast")
def _cast(np: Any, x: Any, dtype: Any, **kwargs: Any) -> Any:
    """Implementation of _cast.
    Args:
        np: The numpy module.
        x: The array.
        dtype: The dtype.
        **kwargs: keyword args.
    Returns:
        The result.
    """
    return np.asarray(x).astype(dtype)  # pragma: no cover


@numpy_eager_registry.register("Angle")
def _angle(np: Any, z: Any, deg: bool = False, **kwargs: Any) -> Any:
    """Implementation of _angle.
    Args:
        np: The numpy module.
        z: The input.
        deg: Degrees.
        **kwargs: keyword args.
    Returns:
        The result.
    """
    return np.angle(z, deg=deg)


@numpy_eager_registry.register("Append")
def _append(np: Any, arr: Any, values: Any, axis: Any = None, **kwargs: Any) -> Any:
    """Implementation of _append.
    Args:
        np: The numpy module.
        arr: The array.
        values: The values.
        axis: The axis.
        **kwargs: keyword args.
    Returns:
        The result.
    """
    return np.append(arr, values, axis=axis)


@numpy_eager_registry.register("Average")
def _average(
    np: Any,
    a: Any,
    axis: Any = None,
    weights: Any = None,
    returned: bool = False,
    keepdims: bool = False,
    **kwargs: Any,
) -> Any:
    """Implementation of _average.
    Args:
        np: The numpy module.
        a: The array.
        axis: The axis.
        weights: The weights.
        returned: Whether to return sum of weights.
        keepdims: Whether to keep dims.
        **kwargs: keyword args.
    Returns:
        The result.
    """
    return np.average(  # pragma: no cover
        a, axis=axis, weights=weights, returned=returned, keepdims=keepdims
    )


@numpy_eager_registry.register("Block")
def _block(np: Any, arrays: Any, **kwargs: Any) -> Any:
    """Implementation of _block.
    Args:
        np: The numpy module.
        arrays: The arrays.
        **kwargs: keyword args.
    Returns:
        The result.
    """
    return np.block(arrays)


@numpy_eager_registry.register("Atleast1d")
def _atleast_1d(np: Any, *args: Any, **kwargs: Any) -> Any:
    return np.atleast_1d(*args)  # pragma: no cover


@numpy_eager_registry.register("Atleast2d")
def _atleast_2d(np: Any, *args: Any, **kwargs: Any) -> Any:
    return np.atleast_2d(*args)  # pragma: no cover


@numpy_eager_registry.register("Atleast3d")
def _atleast_3d(np: Any, *args: Any, **kwargs: Any) -> Any:
    return np.atleast_3d(*args)  # pragma: no cover


@numpy_eager_registry.register("ApplyAlongAxis")
def _apply_along_axis(
    np: Any, func1d: Any, axis: int, arr: Any, *args: Any, **kwargs: Any
) -> Any:
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("ApplyOverAxes")
def _apply_over_axes(np: Any, func: Any, a: Any, axes: Any, **kwargs: Any) -> Any:
    return np.apply_over_axes(func, a, axes)  # pragma: no cover


@numpy_eager_registry.register("ArgPartition")
def _argpartition(np: Any, a: Any, kth: Any, axis: int = -1, **kwargs: Any) -> Any:
    return np.argpartition(a, kth, axis=axis)  # pragma: no cover


@numpy_eager_registry.register("ArgWhere")
def _argwhere(np: Any, a: Any, **kwargs: Any) -> Any:
    return np.argwhere(a)  # pragma: no cover


@numpy_eager_registry.register("Choose")
def _choose(
    np: Any, a: Any, choices: Any, out: Any = None, mode: str = "raise", **kwargs: Any
) -> Any:
    return np.choose(a, choices, out=out, mode=mode)


@numpy_eager_registry.register("ColumnStack")
def _column_stack(np: Any, tup: Any, **kwargs: Any) -> Any:
    return np.column_stack(tup)  # pragma: no cover


@numpy_eager_registry.register("Compress")
def _compress(
    np: Any, condition: Any, a: Any, axis: Any = None, out: Any = None, **kwargs: Any
) -> Any:
    return np.compress(condition, a, axis=axis, out=out)


@numpy_eager_registry.register("Convolve")
def _convolve(np: Any, a: Any, v: Any, mode: str = "full", **kwargs: Any) -> Any:
    return np.convolve(a, v, mode=mode)


@numpy_eager_registry.register("CorrCoef")
def _corrcoef(
    np: Any, x: Any, y: Any = None, rowvar: bool = True, **kwargs: Any
) -> Any:
    return np.corrcoef(x, y=y, rowvar=rowvar)  # pragma: no cover


@numpy_eager_registry.register("Correlate")
def _correlate(np: Any, a: Any, v: Any, mode: str = "valid", **kwargs: Any) -> Any:
    return np.correlate(a, v, mode=mode)


@numpy_eager_registry.register("Cov")
def _cov(
    np: Any,
    m: Any,
    y: Any = None,
    rowvar: bool = True,
    bias: bool = False,
    ddof: Any = None,
    fweights: Any = None,
    aweights: Any = None,
    **kwargs: Any,
) -> Any:
    return np.cov(
        m,
        y=y,
        rowvar=rowvar,
        bias=bias,
        ddof=ddof,
        fweights=fweights,
        aweights=aweights,
    )


@numpy_eager_registry.register("ArrayEquiv")
def _array_equiv(np: Any, a1: Any, a2: Any, **kwargs: Any) -> Any:
    return np.array_equiv(a1, a2)  # pragma: no cover


@numpy_eager_registry.register("ArrayRepr")
def _array_repr(
    np: Any,
    arr: Any,
    max_line_width: Any = None,
    precision: Any = None,
    suppress_small: Any = None,
    **kwargs: Any,
) -> Any:
    return np.array_repr(  # pragma: no cover
        arr,
        max_line_width=max_line_width,
        precision=precision,
        suppress_small=suppress_small,
    )


@numpy_eager_registry.register("ArrayStr")
def _array_str(
    np: Any,
    arr: Any,
    max_line_width: Any = None,
    precision: Any = None,
    suppress_small: Any = None,
    **kwargs: Any,
) -> Any:
    return np.array_str(  # pragma: no cover
        arr,
        max_line_width=max_line_width,
        precision=precision,
        suppress_small=suppress_small,
    )


@numpy_eager_registry.register("Bartlett")
def _bartlett(np: Any, M: int, **kwargs: Any) -> Any:
    return np.bartlett(M)


@numpy_eager_registry.register("BitwiseCount")
def _bitwise_count(np: Any, x: Any, **kwargs: Any) -> Any:
    return (
        np.bitwise_count(x)
        if hasattr(np, "bitwise_count")
        else np.vectorize(lambda i: bin(i).count("1"))(x)
    )


@numpy_eager_registry.register("Blackman")
def _blackman(np: Any, M: int, **kwargs: Any) -> Any:
    return np.blackman(M)


@numpy_eager_registry.register("BroadcastArrays")
def _broadcast_arrays(np: Any, *args: Any, **kwargs: Any) -> Any:
    return np.broadcast_arrays(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("CanCast")
def _can_cast(
    np: Any, from_: Any, to: Any, casting: str = "safe", **kwargs: Any
) -> Any:
    return np.can_cast(from_, to, casting=casting)  # pragma: no cover


@numpy_eager_registry.register("Delete")
def _delete(np: Any, arr: Any, obj: Any, axis: Any = None, **kwargs: Any) -> Any:
    return np.delete(arr, obj, axis=axis)


@numpy_eager_registry.register("DiagIndices")
def _diag_indices(np: Any, n: int, ndim: int = 2, **kwargs: Any) -> Any:
    return np.diag_indices(n, ndim=ndim)


@numpy_eager_registry.register("DiagIndicesFrom")
def _diag_indices_from(np: Any, arr: Any, **kwargs: Any) -> Any:
    return np.diag_indices_from(arr)


@numpy_eager_registry.register("Diagflat")
def _diagflat(np: Any, v: Any, k: int = 0, **kwargs: Any) -> Any:
    return np.diagflat(v, k=k)


@numpy_eager_registry.register("Diagonal")
def _diagonal(
    np: Any, a: Any, offset: int = 0, axis1: int = 0, axis2: int = 1, **kwargs: Any
) -> Any:
    return np.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)


@numpy_eager_registry.register("Diff")
def _diff(
    np: Any,
    a: Any,
    n: int = 1,
    axis: int = -1,
    prepend: Any = None,
    append: Any = None,
    **kwargs: Any,
) -> Any:
    return np.diff(
        a,
        n=n,
        axis=axis,
        prepend=prepend if prepend is not None else getattr(np, "_NoValue", None),
        append=append if append is not None else getattr(np, "_NoValue", None),
    )


@numpy_eager_registry.register("Digitize")
def _digitize(np: Any, x: Any, bins: Any, right: bool = False, **kwargs: Any) -> Any:
    return np.digitize(x, bins, right=right)


@numpy_eager_registry.register("Ediff1d")
def _ediff1d(
    np: Any, ary: Any, to_end: Any = None, to_begin: Any = None, **kwargs: Any
) -> Any:
    # pragma: no cover
    return np.ediff1d(ary, to_end=to_end, to_begin=to_begin)  # pragma: no cover


@numpy_eager_registry.register("EinsumPath")
def _einsum_path(np: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.einsum_path(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Extract")
def _extract(np: Any, condition: Any, arr: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.extract(condition, arr)  # pragma: no cover


@numpy_eager_registry.register("Fabs")
def _fabs(np: Any, x: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.fabs(x)  # pragma: no cover


@numpy_eager_registry.register("fill_diagonal")
def _fill_diagonal(np: Any, a: Any, val: Any, wrap: bool = False, **kwargs: Any) -> Any:
    # pragma: no cover
    import copy

    a_copy = copy.deepcopy(a)
    a_data = getattr(a_copy, "data", a_copy)
    val_data = getattr(val, "data", val)
    np.fill_diagonal(a_data, val_data, wrap=wrap)
    return a_copy


@numpy_eager_registry.register("Flatnonzero")
def _flatnonzero(np: Any, a: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.flatnonzero(a)  # pragma: no cover


@numpy_eager_registry.register("Flip")
def _flip(np: Any, m: Any, axis: Any = None, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.flip(m, axis=axis)  # pragma: no cover


@numpy_eager_registry.register("Fliplr")
def _fliplr(np: Any, m: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.fliplr(m)  # pragma: no cover


@numpy_eager_registry.register("Flipud")
def _flipud(np: Any, m: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.flipud(m)  # pragma: no cover


@numpy_eager_registry.register("Reverse")
def _reverse(np: Any, m: Any, dims: Any = None, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.flip(m, axis=dims)


@numpy_eager_registry.register("FromDlpack")
def _from_dlpack(np: Any, x: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    # pragma: no cover
    return np.from_dlpack(x)  # pragma: no cover


@numpy_eager_registry.register("Fromfunction")
def _fromfunction(np: Any, function: Any, shape: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    # pragma: no cover
    return np.fromfunction(function, shape, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Fromiter")
def _fromiter(np: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    # pragma: no cover
    return np.fromiter(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Frompyfunc")
def _frompyfunc(np: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    # pragma: no cover
    return np.frompyfunc(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Fromstring")
def _fromstring(np: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    # pragma: no cover
    return np.fromstring(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Geomspace")
def _geomspace(np: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    # pragma: no cover
    return np.geomspace(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("GetPrintoptions")
def _get_printoptions(np: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    # pragma: no cover
    return np.get_printoptions(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Gradient")
def _gradient(np: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    # pragma: no cover
    return np.gradient(
        *args, **{k: v for k, v in kwargs.items() if v is not None}
    )  # pragma: no cover


@numpy_eager_registry.register("Hamming")
def _hamming(np: Any, M: int, **kwargs: Any) -> Any:
    # pragma: no cover
    # pragma: no cover
    return np.hamming(M)


@numpy_eager_registry.register("Hanning")
def _hanning(np: Any, M: int, **kwargs: Any) -> Any:
    # pragma: no cover
    # pragma: no cover
    return np.hanning(M)


@numpy_eager_registry.register("Histogram")
def _histogram(np: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    # pragma: no cover
    return np.histogram(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Histogram2d")
def _histogram2d(np: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    # pragma: no cover
    return np.histogram2d(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("HistogramBinEdges")
def _histogram_bin_edges(np: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    # pragma: no cover
    return np.histogram_bin_edges(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Histogramdd")
def _histogramdd(np: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    # pragma: no cover
    return np.histogramdd(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("I0")
def _i0(np: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    # pragma: no cover
    return np.i0(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Indices")
def _indices(np: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    # pragma: no cover
    return np.indices(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Insert")
def _insert(np: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    # pragma: no cover
    return np.insert(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Interp")
def _interp(np: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    # pragma: no cover
    return np.interp(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Intersect1d")
def _intersect1d(np: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    # pragma: no cover
    return np.intersect1d(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Iscomplex")
def _iscomplex(np: Any, x: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.iscomplex(x)  # pragma: no cover


@numpy_eager_registry.register("Iscomplexobj")
def _iscomplexobj(np: Any, x: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.iscomplexobj(x)  # pragma: no cover


@numpy_eager_registry.register("Isdtype")
def _isdtype(np: Any, dtype: Any, kind: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    if hasattr(np, "isdtype"):  # pragma: no cover
        return np.isdtype(dtype, kind)  # pragma: no cover
    return np.issubdtype(dtype, kind)  # Fallback for older numpy  # pragma: no cover


@numpy_eager_registry.register("Isin")
def _isin(
    np: Any,
    element: Any,
    test_elements: Any,
    assume_unique: bool = False,
    invert: bool = False,
    **kwargs: Any,
) -> Any:
    # pragma: no cover
    return np.isin(
        element, test_elements, assume_unique=assume_unique, invert=invert
    )  # pragma: no cover


@numpy_eager_registry.register("Isreal")
def _isreal(np: Any, x: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.isreal(x)  # pragma: no cover


@numpy_eager_registry.register("Isrealobj")
def _isrealobj(np: Any, x: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.isrealobj(x)  # pragma: no cover


@numpy_eager_registry.register("Isscalar")
def _isscalar(np: Any, element: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.isscalar(element)  # pragma: no cover


@numpy_eager_registry.register("Issubdtype")
def _issubdtype(np: Any, arg1: Any, arg2: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.issubdtype(arg1, arg2)  # pragma: no cover


@numpy_eager_registry.register("Iterable")
def _iterable(np: Any, y: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.iterable(y)  # pragma: no cover


@numpy_eager_registry.register("Ix")
def _ix_(np: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.ix_(*args)  # pragma: no cover


@numpy_eager_registry.register("Kaiser")
def _kaiser(np: Any, M: int, beta: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    beta_data = getattr(beta, "data", beta)
    return np.kaiser(
        M,
        beta_data.item()
        if hasattr(beta_data, "item") and callable(beta_data.item)
        else beta_data,
    )


@numpy_eager_registry.register("Kron")
def _kron(np: Any, a: Any, b: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.kron(a, b)  # pragma: no cover


@numpy_eager_registry.register("Lexsort")
def _lexsort(np: Any, keys: Any, axis: int = -1, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.lexsort(keys, axis=axis)  # pragma: no cover


@numpy_eager_registry.register("Load")
def _load(np: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.load(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("MaskIndices")
def _mask_indices(np: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.mask_indices(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("MatrixTranspose")
def _matrix_transpose(np: Any, x: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    return (  # pragma: no cover
        np.matrix_transpose(x) if hasattr(np, "matrix_transpose") else np.transpose(x)
    )


@numpy_eager_registry.register("Median")
def _median(
    np: Any,
    a: Any,
    axis: Any = None,
    out: Any = None,
    overwrite_input: bool = False,
    keepdims: bool = False,
    **kwargs: Any,
) -> Any:
    # pragma: no cover
    return np.median(  # pragma: no cover
        a, axis=axis, out=out, overwrite_input=overwrite_input, keepdims=keepdims
    )


@numpy_eager_registry.register("Modf")
def _modf(np: Any, x: Any, out: Any = None, **kwargs: Any) -> Any:
    # pragma: no cover
    return np.modf(x, out=out) if out is not None else np.modf(x)  # pragma: no cover


@numpy_eager_registry.register("nanargmax")
def _nanargmax(np: Any, a: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    a = getattr(a, "data", a)
    if kwargs.get("where") is None:
        kwargs.pop("where", None)
    return getattr(np, "nanargmax")(a, *args, **kwargs)


@numpy_eager_registry.register("nanargmin")
def _nanargmin(np: Any, a: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    a = getattr(a, "data", a)
    if kwargs.get("where") is None:
        kwargs.pop("where", None)
    return getattr(np, "nanargmin")(a, *args, **kwargs)


@numpy_eager_registry.register("nancumprod")
def _nancumprod(np: Any, a: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    a = getattr(a, "data", a)
    if kwargs.get("where") is None:
        kwargs.pop("where", None)
    return getattr(np, "nancumprod")(a, *args, **kwargs)


@numpy_eager_registry.register("nancumsum")
def _nancumsum(np: Any, a: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    a = getattr(a, "data", a)
    if kwargs.get("where") is None:
        kwargs.pop("where", None)
    return getattr(np, "nancumsum")(a, *args, **kwargs)


@numpy_eager_registry.register("nanmean")
def _nanmean(np: Any, a: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    a = getattr(a, "data", a)
    if kwargs.get("where") is None:
        kwargs.pop("where", None)
    return getattr(np, "nanmean")(a, *args, **kwargs)


@numpy_eager_registry.register("nanmedian")
def _nanmedian(np: Any, a: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    a = getattr(a, "data", a)
    if kwargs.get("where") is None:
        kwargs.pop("where", None)
    return getattr(np, "nanmedian")(a, *args, **kwargs)


@numpy_eager_registry.register("nanpercentile")
def _nanpercentile(np: Any, a: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    a = getattr(a, "data", a)
    if kwargs.get("where") is None:
        kwargs.pop("where", None)
    return getattr(np, "nanpercentile")(a, *args, **kwargs)


@numpy_eager_registry.register("nanquantile")
def _nanquantile(np: Any, a: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    a = getattr(a, "data", a)
    if kwargs.get("where") is None:
        kwargs.pop("where", None)
    return getattr(np, "nanquantile")(a, *args, **kwargs)


@numpy_eager_registry.register("nanstd")
def _nanstd(np: Any, a: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    a = getattr(a, "data", a)
    if kwargs.get("where") is None:
        kwargs.pop("where", None)
    return getattr(np, "nanstd")(a, *args, **kwargs)


@numpy_eager_registry.register("nanvar")
def _nanvar(np: Any, a: Any, *args: Any, **kwargs: Any) -> Any:
    # pragma: no cover
    a = getattr(a, "data", a)
    if kwargs.get("where") is None:
        kwargs.pop("where", None)
    return getattr(np, "nanvar")(a, *args, **kwargs)
