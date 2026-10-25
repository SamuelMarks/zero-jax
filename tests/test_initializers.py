"""Tests for zero_jax.nn.initializers."""

from __future__ import annotations

import pytest
from ml_switcheroo_ir import DType

import zero_jax.numpy as jnp
from tests.conftest import assert_allclose
from zero_jax.nn import initializers


class _CustomNamedDType:
    """Helper implementing NamedProtocol."""

    name = "float32"


def test_to_dtype_variants() -> None:
    """Verify _to_dtype handles DType, str, Python types, and NamedProtocol."""
    assert initializers._to_dtype(DType("float32")) == DType("float32")
    assert initializers._to_dtype("float32") == DType("float32")
    assert initializers._to_dtype(float) == DType("float64")
    assert initializers._to_dtype(int) == DType("int64")
    assert initializers._to_dtype(bool) == DType("bool")
    assert initializers._to_dtype(complex) == DType("complex64")
    assert initializers._to_dtype(type("float32", (), {})) == DType("float32")
    assert initializers._to_dtype(_CustomNamedDType()) == DType("float32")
    assert initializers._to_dtype(jnp.float32) == DType("float32")


def test_zeros_ones_constant() -> None:
    """Test zeros, ones, constant."""
    z = initializers.zeros(123, (2, 2))
    assert_allclose(z, [[0.0, 0.0], [0.0, 0.0]])

    o = initializers.ones(123, (2, 2))
    assert_allclose(o, [[1.0, 1.0], [1.0, 1.0]])

    c = initializers.constant(5)(123, (2, 2))
    assert_allclose(c, [[5.0, 5.0], [5.0, 5.0]])


def test_uniform_normal() -> None:
    """Test uniform, normal, and truncated_normal."""
    u = initializers.uniform(0.1)(123, (2, 2))
    assert u.shape == (2, 2)

    n = initializers.normal(0.1)(123, (2, 2))
    assert n.shape == (2, 2)

    tn = initializers.truncated_normal(0.1)(123, (2, 2))
    assert tn.shape == (2, 2)


def test_variance_scaling() -> None:
    """Test variance scaling initializers and error branches."""
    a1 = initializers.variance_scaling(1.0, "fan_in", "truncated_normal")(123, (2, 2))
    assert a1.shape == (2, 2)
    a2 = initializers.variance_scaling(1.0, "fan_out", "normal")(123, (2, 2))
    assert a2.shape == (2, 2)
    a3 = initializers.variance_scaling(1.0, "fan_avg", "uniform")(123, (2, 2))
    assert a3.shape == (2, 2)

    with pytest.raises(ValueError, match="invalid mode"):
        initializers.variance_scaling(1.0, "invalid", "normal")(123, (2, 2))

    with pytest.raises(ValueError, match="invalid distribution"):
        initializers.variance_scaling(1.0, "fan_in", "invalid")(123, (2, 2))


def test_named_initializers() -> None:
    """Test specific named initializers."""
    assert initializers.glorot_uniform()(123, (2, 2)).shape == (2, 2)
    assert initializers.glorot_normal()(123, (2, 2)).shape == (2, 2)
    assert initializers.lecun_uniform()(123, (2, 2)).shape == (2, 2)
    assert initializers.lecun_normal()(123, (2, 2)).shape == (2, 2)
    assert initializers.he_uniform()(123, (2, 2)).shape == (2, 2)
    assert initializers.he_normal()(123, (2, 2)).shape == (2, 2)

    assert initializers.xavier_uniform == initializers.glorot_uniform
    assert initializers.xavier_normal == initializers.glorot_normal
    assert initializers.kaiming_uniform == initializers.he_uniform
    assert initializers.kaiming_normal == initializers.he_normal


def test_orthogonal() -> None:
    """Test orthogonal initializers verify Q^T Q = I."""
    o = initializers.orthogonal()(123, (3, 3))
    assert o.shape == (3, 3)
    arr = jnp.asarray(o)
    assert_allclose(jnp.matmul(arr.T, arr), jnp.eye(3), atol=1e-5)

    # Test rectangular (rows < cols)
    o_rect = initializers.orthogonal()(123, (3, 4))
    assert o_rect.shape == (3, 4)
    arr_rect = jnp.asarray(o_rect)
    assert_allclose(jnp.matmul(arr_rect, arr_rect.T), jnp.eye(3), atol=1e-5)

    # Test other column axis
    o_axis = initializers.orthogonal(column_axis=0)(123, (3, 4))
    assert o_axis.shape == (3, 4)

    with pytest.raises(ValueError, match="requires at least a 2D shape"):
        initializers.orthogonal()(123, (3,))


def test_delta_orthogonal() -> None:
    """Test delta orthogonal convolutional kernel initialization."""
    do = initializers.delta_orthogonal()(123, (3, 3, 2, 2))
    assert do.shape == (3, 3, 2, 2)
    arr = jnp.asarray(do)
    assert_allclose(arr[0, 0], jnp.zeros((2, 2)), atol=1e-6)
    center = jnp.asarray(arr[1, 1])
    assert_allclose(jnp.matmul(center.T, center), jnp.eye(2), atol=1e-5)

    with pytest.raises(ValueError, match="requires a 3D, 4D, or 5D shape"):
        initializers.delta_orthogonal()(123, (3,))


def test_compute_fans() -> None:
    """Test _compute_fans across 0D, 1D, and 4D shapes."""
    assert initializers._compute_fans(()) == (1, 1)
    assert initializers._compute_fans((8,)) == (8, 8)
    assert initializers._compute_fans((3, 3, 4, 5), in_axis=-2, out_axis=-1) == (36, 45)
    assert initializers.variance_scaling(1.0, "fan_in", "normal", in_axis=1, out_axis=2, batch_axis=(0,))(123, (2, 3, 4, 5)).shape == (2, 3, 4, 5)
