"""Pytest configuration and shared fixtures for zero-jax tests."""

from __future__ import annotations

import math
from collections.abc import Iterator

import ml_switcheroo_compiler
import pytest
from ml_switcheroo_compiler.core.tensor import Tensor

from zero_jax._types import ArrayLikeSpec, ArrayProtocol, PyTree, VariadicCallable
from zero_jax.numpy.lax_numpy import ndarray


def _flatten_to_scalars(val: ArrayLikeSpec) -> list[float | int | bool | complex]:
    """Flatten an array, tensor, sequence, or scalar into a flat list of Python scalars.

    Args:
        val: Input array-like or scalar value.

    Returns:
        list[float | int | bool | complex]: Flat list of Python scalar values.
    """
    if isinstance(val, (ndarray, Tensor)):
        return _flatten_to_scalars(val.tolist())
    if isinstance(val, ArrayProtocol):
        return _flatten_to_scalars(val.__array__().tolist())
    if isinstance(val, (list, tuple)):
        out: list[float | int | bool | complex] = []
        for item in val:
            out.extend(_flatten_to_scalars(item))
        return out
    return [val if isinstance(val, (bool, int, float, complex)) else float(val)]


def _infer_shape(val: ArrayLikeSpec) -> tuple[int, ...]:
    """Infer the shape tuple of an array-like object.

    Args:
        val: Input array-like object.

    Returns:
        tuple[int, ...]: Shape tuple.
    """
    return tuple(int(d) for d in val.shape) if isinstance(val, (ndarray, Tensor, ArrayProtocol)) else ()


def _scalars_close(
    a: float | int | bool | complex,
    b: float | int | bool | complex,
    rtol: float,
    atol: float,
    equal_nan: bool,
) -> bool:
    """Compare two scalar values for numerical closeness.

    Args:
        a: First scalar value.
        b: Second scalar value.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        equal_nan: Whether ``NaN`` values compare equal.

    Returns:
        bool: ``True`` if ``a`` and ``b`` are close within tolerance.
    """
    ca = complex(a)
    cb = complex(b)
    return (equal_nan and math.isnan(ca.real) and math.isnan(cb.real)) or ca == cb or abs(ca - cb) <= atol + rtol * abs(cb)


def assert_allclose(
    actual: ArrayLikeSpec,
    expected: ArrayLikeSpec,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    equal_nan: bool = True,
) -> None:
    """Assert that two array-like objects or scalars are elementwise numerically close.

    Args:
        actual: Actual value produced by ``zero_jax``.
        expected: Expected reference value.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        equal_nan: Whether ``NaN`` values at matching positions compare equal.

    Raises:
        AssertionError: If shapes or element values differ beyond tolerance.
    """
    if isinstance(actual, (ndarray, Tensor, ArrayProtocol)) and isinstance(expected, (ndarray, Tensor, ArrayProtocol)):
        s_act = _infer_shape(actual)
        s_exp = _infer_shape(expected)
        assert s_act == s_exp, f"Shape mismatch: {s_act} vs {s_exp}"
    flat_act = _flatten_to_scalars(actual)
    flat_exp = _flatten_to_scalars(expected)
    if len(flat_act) != len(flat_exp):
        flat_exp = flat_exp * len(flat_act)
    for idx, (a, b) in enumerate(zip(flat_act, flat_exp)):
        assert _scalars_close(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan), f"Mismatch at flat index {idx}: actual={a!r}, expected={b!r}"


def assert_array_equal(actual: ArrayLikeSpec, expected: ArrayLikeSpec) -> None:
    """Assert that two array-like objects have identical shapes and element values.

    Args:
        actual: Actual value produced by ``zero_jax``.
        expected: Expected reference value.

    Raises:
        AssertionError: If shapes or element values differ.
    """
    assert_allclose(actual, expected, rtol=0.0, atol=0.0, equal_nan=True)


@pytest.fixture(autouse=True)
def reset_tracing_state() -> Iterator[None]:
    """Reset the global tracing state before and after each test.

    Yields:
        None: Test execution context.
    """
    from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer

    from zero_jax._config_internal import config as _zcfg

    _tracer.is_tracing = False
    _tracer.active_graph = None
    _zcfg.update("jax_enable_x64", False)
    yield
    _tracer.is_tracing = False
    _tracer.active_graph = None
    _zcfg.update("jax_enable_x64", False)


@pytest.fixture(autouse=True)
def switcheroo_config() -> Iterator[None]:
    """Ensure eager mode is active by default during each test.

    Yields:
        None: Test execution context.
    """
    with ml_switcheroo_compiler.core.EagerMode():
        yield


@pytest.fixture
def check_allclose() -> VariadicCallable[None]:
    """Return a helper function comparing ``zero_jax`` outputs against reference outputs.

    Returns:
        VariadicCallable[None]: Comparison function ``_check(zero_val, ref_val, rtol, atol)``.
    """

    def _check(
        zero_val: PyTree,
        ref_val: PyTree,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> None:
        """Recursively assert that ``zero_val`` and ``ref_val`` are numerically close.

        Args:
            zero_val: Output produced by ``zero_jax``.
            ref_val: Output produced by reference implementation.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
        """
        if isinstance(ref_val, (tuple, list)) and isinstance(zero_val, (tuple, list)):
            assert len(zero_val) == len(ref_val)
            for z, r in zip(zero_val, ref_val):
                _check(z, r, rtol, atol)
        elif isinstance(ref_val, dict) and isinstance(zero_val, dict):
            assert set(zero_val.keys()) == set(ref_val.keys())
            for k in ref_val:
                _check(zero_val[k], ref_val[k], rtol, atol)
        else:
            assert_allclose(zero_val, ref_val, rtol=rtol, atol=atol)

    return _check
