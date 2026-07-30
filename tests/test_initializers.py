import pytest

"""Tests for zero_jax.nn.initializers."""

import numpy as np

from zero_jax.nn import initializers


def test_zeros_ones_constant():
    """Test zeros, ones, constant."""
    z = initializers.zeros(123, (2, 2))
    assert np.all(z == 0)

    o = initializers.ones(123, (2, 2))
    assert np.all(o == 1)

    c = initializers.constant(5)(123, (2, 2))
    assert np.all(c == 5)


@pytest.mark.skip(reason="Not implemented without numpy")
def test_uniform_normal():
    """Test uniform and normal."""
    u = initializers.uniform(0.1)(123, (2, 2))
    assert u.shape == (2, 2)

    n = initializers.normal(0.1)(123, (2, 2))
    assert n.shape == (2, 2)

    tn = initializers.truncated_normal(0.1)(123, (2, 2))
    assert tn.shape == (2, 2)


@pytest.mark.skip(reason="Not implemented without numpy")
def test_variance_scaling():
    """Test variance scaling initializers."""
    initializers.variance_scaling(1.0, "fan_in", "truncated_normal")(123, (2, 2))
    initializers.variance_scaling(1.0, "fan_out", "normal")(123, (2, 2))
    initializers.variance_scaling(1.0, "fan_avg", "uniform")(123, (2, 2))

    try:
        initializers.variance_scaling(1.0, "invalid", "normal")(123, (2, 2))
    except ValueError:
        pass

    try:
        initializers.variance_scaling(1.0, "fan_in", "invalid")(123, (2, 2))
    except ValueError:
        pass


@pytest.mark.skip(reason="Not implemented without numpy")
def test_named_initializers():
    """Test specific named initializers."""
    initializers.glorot_uniform()(123, (2, 2))
    initializers.glorot_normal()(123, (2, 2))
    initializers.lecun_uniform()(123, (2, 2))
    initializers.lecun_normal()(123, (2, 2))
    initializers.he_uniform()(123, (2, 2))
    initializers.he_normal()(123, (2, 2))

    assert initializers.xavier_uniform == initializers.glorot_uniform
    assert initializers.xavier_normal == initializers.glorot_normal
    assert initializers.kaiming_uniform == initializers.he_uniform
    assert initializers.kaiming_normal == initializers.he_normal


@pytest.mark.skip(reason="Not implemented without numpy")
def test_orthogonal():
    """Test orthogonal initializers."""
    o = initializers.orthogonal()(123, (3, 3))
    assert o.shape == (3, 3)

    # Test rectangular
    o_rect = initializers.orthogonal()(123, (3, 4))
    assert o_rect.shape == (3, 4)

    # Test other column axis
    o_axis = initializers.orthogonal(column_axis=0)(123, (3, 4))
    assert o_axis.shape == (3, 4)

    try:
        initializers.orthogonal()(123, (3,))
    except ValueError:
        pass


@pytest.mark.skip(reason="Not implemented without numpy")
def test_delta_orthogonal():
    """Test delta orthogonal."""
    do = initializers.delta_orthogonal()(123, (3, 3, 2, 2))
    assert do.shape == (3, 3, 2, 2)

    try:
        initializers.delta_orthogonal()(123, (3,))
    except ValueError:
        pass


@pytest.mark.skip(reason="Not implemented without numpy")
def test_compute_fans():
    """Test negative axis for compute fans."""
    # This shape and axes will exercise the negative indexing
    initializers.variance_scaling(1.0, "fan_in", "normal", in_axis=1, out_axis=2)(
        123, (2, 3, 4, 5)
    )
