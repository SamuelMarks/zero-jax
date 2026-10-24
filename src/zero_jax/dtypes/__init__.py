"""Frontend API routing for jax.dtypes."""

from __future__ import annotations

from ml_switcheroo_ir import DType

import zero_jax._compiler_proxy_ops as _ops
from zero_jax._types import ArrayLikeSpec, ArrayProtocol, ConfigValue, DTypeSpec, NamedProtocol

_FLOAT_INFO: dict[str, tuple[int, float, float, float]] = {
    "bfloat16": (16, 0.0078125, -3.3895313892515355e38, 3.3895313892515355e38),
    "float16": (16, 0.0009765625, -65504.0, 65504.0),
    "float32": (32, 1.1920928955078125e-07, -3.4028234663852886e38, 3.4028234663852886e38),
    "float64": (64, 2.220446049250313e-16, -1.7976931348623157e308, 1.7976931348623157e308),
}

_INT_INFO: dict[str, tuple[int, int, int]] = {
    "int4": (4, -8, 7),
    "uint4": (4, 0, 15),
    "int8": (8, -128, 127),
    "uint8": (8, 0, 255),
    "int16": (16, -32768, 32767),
    "uint16": (16, 0, 65535),
    "int32": (32, -2147483648, 2147483647),
    "uint32": (32, 0, 4294967295),
    "int64": (64, -9223372036854775808, 9223372036854775807),
    "uint64": (64, 0, 18446744073709551615),
}


def _resolve_dtype_name(dtype: ConfigValue = None) -> str:
    """Resolve a dtype-like object to its normalized lowercase string name.

    Args:
        dtype: Input dtype specification or scalar class.

    Returns:
        str: Normalized dtype name string.
    """
    if dtype is None:
        return "float32"
    if isinstance(dtype, DType):
        return str(dtype.value).lower()
    if isinstance(dtype, NamedProtocol):
        return dtype.name.lower()
    if isinstance(dtype, str):
        return dtype.lower()
    if dtype is float:
        return "float32"
    if dtype is int:
        return "int32"
    if dtype is bool:
        return "bool"
    if dtype is complex:
        return "complex64"
    return str(dtype).lower()


class _ExtendedScalar:
    """Base scalar type representation for extended JAX dtypes."""

    dtype_name: str = "extended"

    def __init__(
        self,
        value: ConfigValue = 0,
        id: ConfigValue = None,
        name: ConfigValue = None,
    ) -> None:
        """Initialize an extended scalar value.

        Args:
            value: Numerical or configuration value.
            id: Optional identifier metadata.
            name: Optional name override.
        """
        self.value = value
        self.id = id
        self.name = str(name) if name is not None else self.dtype_name
        self.dtype = DType(self.dtype_name) if self.dtype_name in ("bfloat16", "float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool") else self.dtype_name


class bfloat16(_ExtendedScalar):
    """bfloat16 floating-point scalar type."""

    dtype_name = "bfloat16"


class float8_e4m3b11fnuz(_ExtendedScalar):
    """8-bit floating-point scalar type (E4M3B11FNUZ)."""

    dtype_name = "float8_e4m3b11fnuz"


class float8_e4m3fn(_ExtendedScalar):
    """8-bit floating-point scalar type (E4M3FN)."""

    dtype_name = "float8_e4m3fn"


class float8_e4m3fnuz(_ExtendedScalar):
    """8-bit floating-point scalar type (E4M3FNUZ)."""

    dtype_name = "float8_e4m3fnuz"


class float8_e5m2(_ExtendedScalar):
    """8-bit floating-point scalar type (E5M2)."""

    dtype_name = "float8_e5m2"


class float8_e5m2fnuz(_ExtendedScalar):
    """8-bit floating-point scalar type (E5M2FNUZ)."""

    dtype_name = "float8_e5m2fnuz"


class int4(_ExtendedScalar):
    """4-bit signed integer scalar type."""

    dtype_name = "int4"


class uint4(_ExtendedScalar):
    """4-bit unsigned integer scalar type."""

    dtype_name = "uint4"


class extended(_ExtendedScalar):
    """Scalar class for extended dtypes."""

    dtype_name = "extended"


class prng_key(_ExtendedScalar):
    """Scalar class for PRNG Key dtypes."""

    dtype_name = "prng_key"


class finfo:
    """Machine limits for floating-point types."""

    def __init__(
        self,
        dtype: ConfigValue = "float32",
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize floating-point machine limit properties for ``dtype``.

        Args:
            dtype: Floating-point dtype or scalar value.
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
        """
        dtype_str = _resolve_dtype_name(dtype)
        bits, eps, min_val, max_val = _FLOAT_INFO.get(dtype_str, _FLOAT_INFO["float32"])
        self.dtype = dtype_str
        self.bits = bits
        self.eps = eps
        self.min = min_val
        self.max = max_val
        self.tiny = float(2.0 ** (-126 if bits <= 32 else -1022))
        self.id = id
        self.name = name
        self.value = value


class iinfo:
    """Machine limits for integer types."""

    def __init__(
        self,
        int_type: ConfigValue = "int32",
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize integer machine limit properties for ``int_type``.

        Args:
            int_type: Integer dtype or scalar value.
            id: Optional identifier metadata.
            name: Optional name metadata.
            value: Optional value metadata.
        """
        dtype_str = _resolve_dtype_name(int_type)
        bits, min_val, max_val = _INT_INFO.get(dtype_str, _INT_INFO["int32"])
        self.dtype = dtype_str
        self.bits = bits
        self.min = min_val
        self.max = max_val
        self.id = id
        self.name = name
        self.value = value


def float0(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> str:
    """Return the symbolic tangent dtype ``"float0"`` for integer/boolean primals.

    Args:
        *args: Optional positional arguments.
        **kwargs: Optional keyword arguments.

    Returns:
        str: The ``"float0"`` dtype identifier.
    """
    return "float0"


def canonicalize_dtype(dtype: DTypeSpec = "float32", allow_extended_dtype: bool = False) -> str:
    """Convert a dtype to its canonical JAX dtype string based on x64 configuration.

    Args:
        dtype: Input dtype specification.
        allow_extended_dtype: Whether to allow extended dtypes such as ``prng_key``.

    Returns:
        str: Canonicalized dtype name (e.g. ``"float32"`` or ``"int32"``).
    """
    name = _resolve_dtype_name(dtype)
    if name in ("float64", "double"):
        return "float32"
    if name in ("int64", "long"):
        return "int32"
    if name in ("uint64",):
        return "uint32"
    if name in ("complex128",):
        return "complex64"
    return name


def issubdtype(arg1: DTypeSpec = "float32", arg2: DTypeSpec = "float32") -> bool:
    """Return ``True`` if ``arg1`` is equal to or a subtype of ``arg2`` in the dtype hierarchy.

    Args:
        arg1: First dtype or category name.
        arg2: Second dtype or category name.

    Returns:
        bool: Whether ``arg1`` is a subdtype of ``arg2``.
    """
    n1 = _resolve_dtype_name(arg1)
    n2 = _resolve_dtype_name(arg2)
    if n1 == n2:
        return True
    floating = {"bfloat16", "float16", "float32", "float64", "floating", "inexact", "number"}
    signed_int = {"int4", "int8", "int16", "int32", "int64", "signedinteger", "integer", "number"}
    unsigned_int = {"uint4", "uint8", "uint16", "uint32", "uint64", "unsignedinteger", "integer", "number"}
    complex_types = {"complex64", "complex128", "complexfloating", "inexact", "number"}
    if n1 in floating and n2 in ("floating", "inexact", "number", "generic"):
        return True
    if n1 in signed_int and n2 in ("signedinteger", "integer", "number", "generic"):
        return True
    if n1 in unsigned_int and n2 in ("unsignedinteger", "integer", "number", "generic"):
        return True
    return bool(n1 in complex_types and n2 in ("complexfloating", "inexact", "number", "generic"))


def result_type(*args: ArrayLikeSpec) -> str:
    """Determine the promoted JAX dtype string for a sequence of values or dtypes.

    Args:
        *args: Input arrays, scalars, or dtype specifications.

    Returns:
        str: Promoted canonical dtype name.
    """
    if not args:
        return "float32"
    rank_order = ["bool", "uint8", "int8", "uint16", "int16", "uint32", "int32", "bfloat16", "float16", "float32", "complex64"]
    best_idx = 0
    for item in args:
        if isinstance(item, ArrayProtocol):
            name = _resolve_dtype_name(item.dtype)
        elif isinstance(item, (bool, int, float, complex)):
            name = _resolve_dtype_name(type(item))
        else:
            name = _resolve_dtype_name(item)
        canon = canonicalize_dtype(name)
        if canon in rank_order:
            idx = rank_order.index(canon)
            if idx > best_idx:
                best_idx = idx
    return rank_order[best_idx]


def scalar_type_of(x: ArrayLikeSpec = 0.0) -> type[bool | int | float | complex]:
    """Return the Python scalar type associated with a JAX value or dtype.

    Args:
        x: Input array, scalar, or dtype specification.

    Returns:
        type[bool | int | float | complex]: Corresponding Python scalar type.
    """
    dtype_str = result_type(x)
    if "complex" in dtype_str:
        return complex
    if "float" in dtype_str or dtype_str == "bfloat16":
        return float
    if "int" in dtype_str:
        return int
    return bool


__all__ = [
    "bfloat16",
    "canonicalize_dtype",
    "extended",
    "finfo",
    "float0",
    "float8_e4m3b11fnuz",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "iinfo",
    "int4",
    "issubdtype",
    "prng_key",
    "result_type",
    "scalar_type_of",
    "uint4",
]
