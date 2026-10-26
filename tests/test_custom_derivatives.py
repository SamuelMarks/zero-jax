"""Tests for zero_jax.custom_derivatives."""

from __future__ import annotations

import zero_jax.custom_derivatives as mod
import zero_jax.numpy as jnp
from tests.conftest import assert_allclose


def test_CustomVJPPrimal() -> None:
    """Test CustomVJPPrimal."""
    obj = mod.CustomVJPPrimal(value=jnp.array(3.0), perturbed=False)
    assert_allclose(obj.value, 3.0)
    assert obj.perturbed is False


def test_SymbolicZero() -> None:
    """Test SymbolicZero."""
    obj = mod.SymbolicZero("float32[]")
    assert obj.aval == "float32[]"
    assert obj.value == 0


def test_closure_convert() -> None:
    """Test closure_convert."""
    conv_fn, aux = mod.closure_convert(lambda x: x + 1.0, jnp.array(2.0))
    assert_allclose(conv_fn(jnp.array(2.0)), 3.0)
    assert aux == []


def test_custom_gradient() -> None:
    """Test custom_gradient."""
    cg_fn = mod.custom_gradient(lambda x: (x * 2.0, lambda g: g * 5.0))
    assert_allclose(cg_fn(jnp.array(3.0)), 6.0)


def test_custom_jvp() -> None:
    """Test custom_jvp."""
    obj = mod.custom_jvp()
    assert obj() is None
    cj = mod.custom_jvp(lambda x: x + 2.0)
    rule = lambda p, t: (p[0] + 2.0, t[0])
    assert cj.defjvp(rule) is rule
    assert_allclose(cj(jnp.array(5.0)), 7.0)


def test_custom_vjp() -> None:
    """Test custom_vjp."""
    obj = mod.custom_vjp()
    assert obj() is None
    cv = mod.custom_vjp(lambda x: x * 4.0)
    cv.defvjp(lambda x: (x * 4.0, None), lambda _, g: (g * 4.0,))
    assert_allclose(cv(jnp.array(3.0)), 12.0)


def test_custom_vjp_primal_tree_values() -> None:
    """Test custom_vjp_primal_tree_values."""
    assert mod.custom_vjp_primal_tree_values(42) == 42
    primal = mod.CustomVJPPrimal(99, True)
    assert mod.custom_vjp_primal_tree_values(primal) == 99


def test_linear_call() -> None:
    """Test linear_call."""
    res = mod.linear_call(lambda x: x + 10.0, lambda g: g, jnp.array(5.0))
    assert_allclose(res, 15.0)
