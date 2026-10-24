"""Pytest configuration and shared fixtures for zero-jax tests."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")),
)
import ml_switcheroo_compiler

from zero_jax._types import PyTree, VariadicCallable

try:
    import jax

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


@pytest.fixture(autouse=True)
def reset_tracing_state() -> Iterator[None]:
    """Reset the global tracing state before and after each test.

    Yields:
        None: Test execution context.
    """
    from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer

    _tracer.is_tracing = False
    _tracer.active_graph = None
    yield
    _tracer.is_tracing = False
    _tracer.active_graph = None


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
    """Return a helper function comparing ``zero_jax`` outputs against reference JAX outputs.

    Returns:
        VariadicCallable[None]: Comparison function ``_check(zero_val, ref_val, rtol, atol)``.
    """
    if not HAS_JAX:
        pytest.skip("Official JAX is not installed. Skipping reference tests.")

    def _check(
        zero_val: PyTree,
        ref_val: PyTree,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> None:
        """Recursively assert that ``zero_val`` and ``ref_val`` are numerically close.

        Args:
            zero_val: Output produced by ``zero_jax``.
            ref_val: Output produced by reference ``jax``.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
        """
        if isinstance(ref_val, (jax.Array, np.ndarray, float, int, bool)):
            np.testing.assert_allclose(zero_val, ref_val, rtol=rtol, atol=atol)
        elif isinstance(ref_val, (tuple, list)) and isinstance(zero_val, (tuple, list)):
            assert len(zero_val) == len(ref_val)
            for z, r in zip(zero_val, ref_val):
                _check(z, r, rtol, atol)
        elif isinstance(ref_val, dict) and isinstance(zero_val, dict):
            assert set(zero_val.keys()) == set(ref_val.keys())
            for k in ref_val:
                _check(zero_val[k], ref_val[k], rtol, atol)
        else:
            assert zero_val == ref_val

    return _check
