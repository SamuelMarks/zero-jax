"""Tests for zero_jax.dtypes."""

from __future__ import annotations

from ml_switcheroo_ir import DType

import zero_jax.dtypes as mod
import zero_jax.numpy as jnp


class _NamedDType:
    """Helper implementing NamedProtocol."""

    name = "float16"


def test_extended_scalar_types() -> None:
    """Verify extended scalar classes and attributes."""
    bf = mod.bfloat16(1.5)
    assert bf.value == 1.5
    assert bf.name == "bfloat16"
    assert mod.float8_e4m3b11fnuz(1.0).dtype == "float8_e4m3b11fnuz"
    assert mod.float8_e4m3fn(1.0).dtype == "float8_e4m3fn"
    assert mod.float8_e4m3fnuz(1.0).dtype == "float8_e4m3fnuz"
    assert mod.float8_e5m2(1.0).dtype == "float8_e5m2"
    assert mod.float8_e5m2fnuz(1.0).dtype == "float8_e5m2fnuz"
    assert mod.int4(2).dtype == "int4"
    assert mod.uint4(2).dtype == "uint4"
    assert mod.extended(0).dtype == "extended"
    assert mod.prng_key(0).dtype == "prng_key"


def test_resolve_dtype_name_variants() -> None:
    """Verify _resolve_dtype_name across all supported input representations."""
    assert mod._resolve_dtype_name(None) == "float32"
    assert mod._resolve_dtype_name(DType("float32")) == "float32"
    assert mod._resolve_dtype_name(_NamedDType()) == "float16"
    assert mod._resolve_dtype_name("INT32") == "int32"
    assert mod._resolve_dtype_name(float) == "float32"
    assert mod._resolve_dtype_name(int) == "int32"
    assert mod._resolve_dtype_name(bool) == "bool"
    assert mod._resolve_dtype_name(complex) == "complex64"
    assert mod._resolve_dtype_name(123) == "123"


def test_canonicalize_dtype() -> None:
    """Verify canonicalize_dtype normalizes 64-bit types to 32-bit defaults."""
    assert mod.canonicalize_dtype("float64") == "float32"
    assert mod.canonicalize_dtype("int64") == "int32"
    assert mod.canonicalize_dtype("uint64") == "uint32"
    assert mod.canonicalize_dtype("complex128") == "complex64"
    assert mod.canonicalize_dtype("float16") == "float16"


def test_finfo_and_iinfo() -> None:
    """Verify finfo and iinfo machine limits."""
    f32 = mod.finfo("float32")
    assert f32.bits == 32
    assert f32.eps > 0
    f64 = mod.finfo("float64")
    assert f64.bits == 64

    i32 = mod.iinfo("int32")
    assert i32.bits == 32
    assert i32.min < 0 < i32.max
    i4 = mod.iinfo("int4")
    assert i4.bits == 4
    assert i4.min == -8 and i4.max == 7


def test_issubdtype() -> None:
    """Verify issubdtype hierarchy checks."""
    assert mod.issubdtype("float32", "float32") is True
    assert mod.issubdtype("float32", "floating") is True
    assert mod.issubdtype("int32", "signedinteger") is True
    assert mod.issubdtype("uint8", "unsignedinteger") is True
    assert mod.issubdtype("complex64", "complexfloating") is True
    assert mod.issubdtype("int32", "floating") is False


def test_result_type_and_scalar_type_of() -> None:
    """Verify result_type promotion and scalar_type_of."""
    assert mod.result_type() == "float32"
    arr = jnp.array([1, 2])
    assert mod.result_type(arr, 1.0) == "float32"
    assert mod.result_type(True, 1) == "int32"
    assert mod.result_type("unknown_dtype") == "bool"

    assert mod.scalar_type_of(1.0) is float
    assert mod.scalar_type_of(1) is int
    assert mod.scalar_type_of(True) is bool
    assert mod.scalar_type_of(1 + 2j) is complex
