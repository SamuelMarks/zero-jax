"""NumPy scalar type hierarchy and dtype helpers for zero_jax."""

from __future__ import annotations

import builtins
from collections.abc import Callable

import ml_switcheroo_compiler as _ml_switcheroo_compiler
from ml_switcheroo_compiler import Tensor
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType as _DType
from ml_switcheroo_compiler.core.tensor import TensorConfig as _TensorConfig

import zero_jax._compiler_proxy_ops as ops
from zero_jax._types import (
    ArrayLikeSpec,
    ArrayProtocol,
    ConfigValue,
    DTypeSpec,
    VariadicCallable,
)
from zero_jax.numpy._ndarray import _to_tensor, _wrap, ndarray
from zero_jax.numpy.tensor_utils import to_array as _to_array


class _ExtendedDType(str):
    """Distinct dtype descriptor for extended integer and 8-bit floating-point types."""

    name: str
    storage_dtype: _DType

    def __new__(cls, name: str, storage_dtype: _DType) -> _ExtendedDType:
        """Create a new ``_ExtendedDType`` instance.

        Args:
            name: Canonical extended dtype name.
            storage_dtype: Backing compiler storage ``DType``.

        Returns:
            _ExtendedDType: Initialized extended dtype descriptor.
        """
        obj = super().__new__(cls, storage_dtype.value)
        obj.name = name
        obj.storage_dtype = storage_dtype
        return obj


bool = _DType.Bool
bool_ = _DType.Bool
complex128 = _DType.Complex128
complex64 = _DType.Complex64
complex_ = _DType.Complex128
float16 = _DType.Float16
bfloat16 = _DType.BFloat16
float8_e4m3b11fnuz = _ExtendedDType("float8_e4m3b11fnuz", _DType.Float16)
float8_e4m3fn = _ExtendedDType("float8_e4m3fn", _DType.Float16)
float8_e4m3fnuz = _ExtendedDType("float8_e4m3fnuz", _DType.Float16)
float8_e5m2 = _ExtendedDType("float8_e5m2", _DType.Float16)
float8_e5m2fnuz = _ExtendedDType("float8_e5m2fnuz", _DType.Float16)
float32 = _DType.Float32
float64 = _DType.Float64
float_ = _DType.Float64
int16 = _DType.Int16
int32 = _DType.Int32
int64 = _DType.Int64
int8 = _DType.Int8
int4 = _ExtendedDType("int4", _DType.Int8)
int_ = _DType.Int64
uint16 = _ExtendedDType("uint16", _DType.Int16)
uint32 = _DType.UInt32
uint64 = _ExtendedDType("uint64", _DType.Int64)
uint8 = _DType.UInt8
double = float64
single = float32
csingle = complex64
cdouble = complex128


class generic:
    """generic."""


class number(generic):
    """number."""


class integer(number):
    """integer."""


class signedinteger(integer):
    """signedinteger."""


class unsignedinteger(integer):
    """unsignedinteger."""


class inexact(number):
    """inexact."""


class floating(inexact):
    """floating."""


class complexfloating(inexact):
    """complexfloating."""


class flexible(generic):
    """flexible."""


class character(flexible):
    """character."""


class object_(generic):
    """object_."""


class ComplexWarning(Warning):
    """Warning raised when casting a complex dtype to a real dtype."""

    def __init__(self, message: str = "Casting complex values to real discards the imaginary part") -> None:
        """Initialize ``ComplexWarning``.

        Args:
            message: Warning message string.
        """
        super().__init__(message)
        self.message: str = str(message)


def dtype(
    value: ArrayLikeSpec,
    names: ArrayLikeSpec = None,
    *,
    module: ArrayLikeSpec = None,
    qualname: ArrayLikeSpec = None,
    type: ArrayLikeSpec = None,
    start: int = 1,
) -> ArrayLikeSpec:
    """Create a data type object.

    Args:
        value: Object to be converted to a data type object.
        names: Optional.
        module: Optional.
        qualname: Optional.
        type: Optional.
        start: Optional.

    Returns:
        A new dtype object.
    """
    try:
        if isinstance(value, _DType):
            return value
        name_str = value if isinstance(value, str) else (value.__name__ if isinstance(value, type) else str(value))
        return _DType(name_str.lower())
    except Exception:
        return value


def finfo(dtype: ArrayLikeSpec) -> ArrayLikeSpec:
    """Machine limits for floating point types.

    Args:
        dtype: The kind of floating point data-type.

    Returns:
        Machine limits.
    """
    from zero_jax.dtypes import finfo as _dtypes_finfo

    return _dtypes_finfo(dtype)


def iinfo(int_type: ArrayLikeSpec) -> ArrayLikeSpec:
    """Machine limits for integer types.

    Args:
        int_type: The kind of integer data type.

    Returns:
        Machine limits.
    """
    from zero_jax.dtypes import iinfo as _dtypes_iinfo

    return _dtypes_iinfo(int_type)


def can_cast(from_: ArrayLikeSpec, to: ArrayLikeSpec, casting: str = "safe") -> ArrayLikeSpec:
    """Returns True if cast between data types can occur according to the casting rule.

    Args:
        from_: Source type.
        to: Destination type.
        casting: Casting rule.

    Returns:
        True if cast can occur.
    """
    return _safe_bool(ops.can_cast(from_, to, casting=casting))


def promote_types(type1: ArrayLikeSpec, type2: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``promote_types``.

    Args:
        type1: Input parameter ``type1``.
        type2: Input parameter ``type2``.

    Returns:
        The operation result.
    """
    return _wrap(ops.promote_types(type1, type2))


def result_type(*arrays_and_dtypes: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``result_type``.

    Args:
        *arrays_and_dtypes: Input parameter ``arrays_and_dtypes``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    processed = [_to_tensor(x) if isinstance(x, (ndarray, Tensor, ArrayProtocol)) else x for x in arrays_and_dtypes]
    return _wrap(ops.result_type(*processed))


def issubdtype(arg1: ArrayLikeSpec, arg2: ArrayLikeSpec) -> bool:
    """Execute ``issubdtype``.

    Args:
        arg1: Input parameter ``arg1``.
        arg2: Input parameter ``arg2``.

    Returns:
        The operation result.
    """
    return _safe_bool(ops.issubdtype(arg1, arg2))


def isdtype(dtype: ArrayLikeSpec, kind: ArrayLikeSpec) -> bool:
    """Execute ``isdtype``.

    Args:
        dtype: Input parameter ``dtype``.
        kind: Input parameter ``kind``.

    Returns:
        The operation result.
    """
    return _safe_bool(ops.issubdtype(dtype, kind))


def uint(x: ArrayLikeSpec = 0) -> ArrayLikeSpec:
    """Execute ``uint``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    from zero_jax.numpy._creation import array

    return array(x, dtype=_DType.UInt32)


def uint4(x: ArrayLikeSpec = 0) -> ArrayLikeSpec:
    """Execute ``uint4``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    from zero_jax.numpy._creation import array

    return array(x, dtype=_DType.UInt8)


def astype(x: ArrayLikeSpec, dtype: ArrayLikeSpec, copy: ArrayLikeSpec = False, device: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Copy of the array, cast to a specified type.

    Args:
        x: Input array.
        dtype: Data type to which the array is cast.
        copy: Whether to copy the array.
        device: Device to place the returned array.

    Returns:
        The cast array.
    """
    return _wrap(ops.cast(_to_tensor(x), dtype))


def isscalar(element: ArrayLikeSpec) -> builtins.bool:
    """Execute ``isscalar``.

    Args:
        element: Input parameter ``element``.

    Returns:
        The operation result.
    """
    if isinstance(element, (ndarray, Tensor)):
        return False
    return isinstance(element, (int, float, complex, builtins.bool, str, bytes)) or element is None


def iterable(y: ArrayLikeSpec) -> bool:
    """Execute ``iterable``.

    Args:
        y: Input parameter ``y``.

    Returns:
        The operation result.
    """
    raw = y._tensor.data if isinstance(y, ndarray) else (y.data if isinstance(y, Tensor) else y)
    return _safe_bool(ops.iterable(raw))


def iscomplex(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Returns a bool array, where True if input element is complex.

    Args:
        x: Input array.

    Returns:
        Boolean array.
    """
    return _wrap(ops.iscomplex(_to_tensor(x)))


def iscomplexobj(x: ArrayLikeSpec) -> bool:
    # Using proxy node if possible
    """Execute ``iscomplexobj``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    return _safe_bool(ops.iscomplexobj(_to_tensor(x)))


def isreal(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Returns a bool array, where True if input element is real.

    Args:
        x: Input array.

    Returns:
        Boolean array.
    """
    return _wrap(ops.isreal(_to_tensor(x)))


def isrealobj(x: ArrayLikeSpec) -> bool:
    """Execute ``isrealobj``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    raw = x._tensor.data if isinstance(x, ndarray) else (x.data if isinstance(x, Tensor) else x)
    return _safe_bool(ops.isrealobj(raw))


def _safe_bool(val: ArrayLikeSpec) -> builtins.bool:
    """Safely convert a tensor or scalar to a Python boolean.

    Args:
        val: Input tensor or scalar value.

    Returns:
        bool: Python boolean value.
    """
    if isinstance(val, Tensor):
        return builtins.bool(val.data)
    return builtins.bool(val)


def get_printoptions() -> ArrayLikeSpec:
    """Return the current print options dictionary.

    Returns:
        ArrayLikeSpec: Dictionary of active print formatting options.
    """
    return {
        "edgeitems": 3,
        "threshold": 1000,
        "floatmode": "maxprec",
        "precision": 8,
        "suppress": False,
        "linewidth": 75,
        "nanstr": "nan",
        "infstr": "inf",
        "sign": "-",
        "formatter": None,
        "legacy": False,
    }


def array_repr(
    arr: ArrayLikeSpec,
    max_line_width: ArrayLikeSpec = None,
    precision: ArrayLikeSpec = None,
    suppress_small: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Return the string representation of an array.

    Args:
        arr: Input array.
        max_line_width: Maximum line width.
        precision: Floating point precision.
        suppress_small: Whether to suppress very small numbers.

    Returns:
        String representation.
    """
    return ops.array_repr(
        _to_tensor(arr),
        max_line_width=max_line_width,
        precision=precision,
        suppress_small=suppress_small,
    )


def array_str(
    arr: ArrayLikeSpec,
    max_line_width: ArrayLikeSpec = None,
    precision: ArrayLikeSpec = None,
    suppress_small: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Return a string representation of the data in an array.

    Args:
        arr: Input array.
        max_line_width: Maximum line width.
        precision: Floating point precision.
        suppress_small: Whether to suppress very small numbers.

    Returns:
        String representation.
    """
    return ops.array_str(
        _to_tensor(arr),
        max_line_width=max_line_width,
        precision=precision,
        suppress_small=suppress_small,
    )


def load(
    file: str,
    mmap_mode: str | None = None,
    allow_pickle: builtins.bool = False,
    fix_imports: builtins.bool = True,
    encoding: str = "ASCII",
) -> ArrayLikeSpec:
    """Load arrays or pickled objects from ``.npy``, ``.npz`` or pickled files.

    Args:
        file: File path to read.
        mmap_mode: Optional memory-map mode.
        allow_pickle: Allow loading pickled object arrays.
        fix_imports: Fix Python 2 to Python 3 import names.
        encoding: Encoding to use when reading Python 2 strings.

    Returns:
        ArrayLikeSpec: Loaded array data.
    """
    return _wrap(
        ops.load(
            file,
            mmap_mode=mmap_mode,
            allow_pickle=allow_pickle,
            fix_imports=fix_imports,
            encoding=encoding,
        )
    )


def save(file: ArrayLikeSpec, arr: ArrayLikeSpec, allow_pickle: bool = True, fix_imports: bool = True) -> None:
    """JAX API implementation for save.

    Args:
        file: Input parameter ``file``.
        arr: Input parameter ``arr``.
        allow_pickle: Input parameter ``allow_pickle``.
        fix_imports: Input parameter ``fix_imports``.
    """
    ops.save(file, _to_tensor(arr), allow_pickle, fix_imports)


def savez(file: ArrayLikeSpec, *args: ArrayLikeSpec, **kwds: ArrayLikeSpec) -> None:
    """JAX API implementation for savez.

    Args:
        file: Input parameter ``file``.
        *args: Input parameter ``args``.
        **kwds: Input parameter ``kwds``.
    """
    ops.savez(
        file,
        *[_to_tensor(a) for a in args],
        **{k: _to_tensor(v) for k, v in kwds.items()},
    )


__all__ = [
    "ComplexWarning",
    "_ExtendedDType",
    "_safe_bool",
    "array_repr",
    "array_str",
    "astype",
    "bfloat16",
    "bool",
    "bool_",
    "can_cast",
    "cdouble",
    "character",
    "complex128",
    "complex64",
    "complex_",
    "complexfloating",
    "csingle",
    "double",
    "dtype",
    "finfo",
    "flexible",
    "float16",
    "float32",
    "float64",
    "float8_e4m3b11fnuz",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float_",
    "floating",
    "generic",
    "get_printoptions",
    "iinfo",
    "inexact",
    "int16",
    "int32",
    "int4",
    "int64",
    "int8",
    "int_",
    "integer",
    "iscomplex",
    "iscomplexobj",
    "isdtype",
    "isreal",
    "isrealobj",
    "isscalar",
    "issubdtype",
    "iterable",
    "load",
    "number",
    "object_",
    "promote_types",
    "result_type",
    "save",
    "savez",
    "signedinteger",
    "single",
    "uint",
    "uint16",
    "uint32",
    "uint4",
    "uint64",
    "uint8",
    "unsignedinteger",
]
