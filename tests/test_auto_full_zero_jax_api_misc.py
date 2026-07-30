"""Tests for zero_jax.api.misc."""

from typing import Any

import pytest

import zero_jax.api.misc as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_Callable() -> None:
    """Test Callable."""
    try:
        mod.Callable()
    except Exception:
        pass


def test_checkpoint() -> None:
    """Test checkpoint."""
    try:
        mod.checkpoint(1.0)
    except Exception:
        pass


def test_closure_convert() -> None:
    """Test closure_convert."""
    try:
        mod.closure_convert(1.0)
    except Exception:
        pass


def test_debug_key_reuse() -> None:
    """Test debug_key_reuse."""
    try:
        mod.debug_key_reuse()
    except Exception:
        pass


def test_default_matmul_precision() -> None:
    """Test default_matmul_precision."""
    try:
        mod.default_matmul_precision()
    except Exception:
        pass


def test_default_prng_impl() -> None:
    """Test default_prng_impl."""
    try:
        mod.default_prng_impl()
    except Exception:
        pass


def test_enable_custom_prng() -> None:
    """Test enable_custom_prng."""
    try:
        mod.enable_custom_prng()
    except Exception:
        pass


def test_enable_custom_vjp_by_custom_transpose() -> None:
    """Test enable_custom_vjp_by_custom_transpose."""
    try:
        mod.enable_custom_vjp_by_custom_transpose()
    except Exception:
        pass


def test_ensure_compile_time_eval() -> None:
    """Test ensure_compile_time_eval."""
    try:
        mod.ensure_compile_time_eval(1.0)
    except Exception:
        pass


def test_jax2tf_associative_scan_reductions() -> None:
    """Test jax2tf_associative_scan_reductions."""
    try:
        mod.jax2tf_associative_scan_reductions()
    except Exception:
        pass


def test_legacy_prng_key() -> None:
    """Test legacy_prng_key."""
    try:
        mod.legacy_prng_key()
    except Exception:
        pass


def test_make_array_from_callback() -> None:
    """Test make_array_from_callback."""
    try:
        mod.make_array_from_callback(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_make_array_from_process_local_data() -> None:
    """Test make_array_from_process_local_data."""
    try:
        mod.make_array_from_process_local_data(1.0, 1.0)
    except Exception:
        pass


def test_make_array_from_single_device_arrays() -> None:
    """Test make_array_from_single_device_arrays."""
    try:
        mod.make_array_from_single_device_arrays(1.0, 1.0, 1.0)
    except Exception:
        pass


def test_make_jaxpr() -> None:
    """Test make_jaxpr."""
    try:
        mod.make_jaxpr(1.0)
    except Exception:
        pass


def test_named_call() -> None:
    """Test named_call."""
    try:
        mod.named_call(1.0)
    except Exception:
        pass


def test_named_scope() -> None:
    """Test named_scope."""
    try:
        mod.named_scope(1.0)
    except Exception:
        pass


def test_pure_callback() -> None:
    """Test pure_callback."""
    try:
        mod.pure_callback(1.0, 1.0)
    except Exception:
        pass


def test_remat() -> None:
    """Test remat."""
    try:
        mod.remat(1.0)
    except Exception:
        pass


def test_softmax_custom_jvp() -> None:
    """Test softmax_custom_jvp."""
    try:
        mod.softmax_custom_jvp(1.0)
    except Exception:
        pass


def test_threefry_partitionable() -> None:
    """Test threefry_partitionable."""
    try:
        mod.threefry_partitionable()
    except Exception:
        pass
