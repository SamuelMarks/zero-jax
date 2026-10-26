"""Tests for zero_jax.custom_derivatives."""

from __future__ import annotations

import zero_jax.custom_derivatives as mod
import zero_jax.numpy as jnp
from tests.conftest import assert_allclose


def test_class_CustomVJPPrimal() -> None:
    """Test class CustomVJPPrimal."""
    obj = mod.CustomVJPPrimal(value=1)
    assert obj.value == 1


def test_class_SymbolicZero() -> None:
    """Test class SymbolicZero."""
    obj = mod.SymbolicZero("float32")
    assert obj.aval == "float32"
    assert obj.value == 0


def test_closure_convert() -> None:
    """Test closure_convert."""
    fn, aux = mod.closure_convert(lambda x: x * 2.0, jnp.array(3.0))
    assert_allclose(fn(jnp.array(3.0)), 6.0)
    assert aux == []


def test_custom_gradient() -> None:
    """Test custom_gradient."""
    cg = mod.custom_gradient(lambda x: (x + 1.0, lambda g: g))
    assert_allclose(cg(jnp.array(4.0)), 5.0)


def test_class_custom_jvp() -> None:
    """Test class custom_jvp."""
    obj = mod.custom_jvp(lambda x: x * 3.0)
    assert_allclose(obj(jnp.array(2.0)), 6.0)


def test_custom_jvp_call_jaxpr_p() -> None:
    """Test custom_jvp_call_jaxpr_p."""
    assert mod.custom_jvp_call_jaxpr_p().name == "custom_jvp_call_jaxpr"


def test_custom_jvp_call_p() -> None:
    """Test custom_jvp_call_p."""
    assert mod.custom_jvp_call_p().name == "custom_jvp_call"


def test_class_custom_vjp() -> None:
    """Test class custom_vjp."""
    obj = mod.custom_vjp(lambda x: x - 1.0)
    assert_allclose(obj(jnp.array(5.0)), 4.0)


def test_custom_vjp_call_jaxpr_p() -> None:
    """Test custom_vjp_call_jaxpr_p."""
    assert mod.custom_vjp_call_jaxpr_p().name == "custom_vjp_call_jaxpr"


def test_custom_vjp_call_p() -> None:
    """Test custom_vjp_call_p."""
    assert mod.custom_vjp_call_p().name == "custom_vjp_call"


def test_custom_vjp_primal_tree_values() -> None:
    """Test custom_vjp_primal_tree_values."""
    assert mod.custom_vjp_primal_tree_values(7) == 7


def test_linear_call() -> None:
    """Test linear_call."""
    assert_allclose(mod.linear_call(lambda x: x * 5.0, lambda g: g * 5.0, jnp.array(2.0)), 10.0)
