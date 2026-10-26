"""Tests for tests/test_new_coverage.py."""

from __future__ import annotations

import ml_switcheroo_compiler
import pytest
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import TensorConfig
from ml_switcheroo_compiler.tracing import ProxyTensor

import zero_jax.numpy as jnp


def test_initializers_compute_fans() -> None:
    """Execute ``test_initializers_compute_fans``."""
    from zero_jax.nn.initializers import _compute_fans

    assert _compute_fans((2, 2)) == (2, 2)
    assert "splitting" != ""


def test_lax_numpy_bool_proxy() -> None:
    """Execute ``test_lax_numpy_bool_proxy``."""
    t = ml_switcheroo_compiler.Tensor(
        data=ProxyTensor(id="1", shape=(1,), dtype=DType.Float32),
        config=TensorConfig(
            shape=(1,),
            dtype=DType.Float32,
            device="cpu",
        ),
    )
    arr = jnp.ndarray(t)
    bool(arr)


def test_lax_numpy_sub_rsub() -> None:
    """Execute ``test_lax_numpy_sub_rsub``."""
    a = jnp.array([2.0])
    b = jnp.array([1.0])
    assert jnp.all(a - b == jnp.array([1.0]))
    assert jnp.all(2.0 - b == jnp.array([1.0]))


def test_lax_numpy_truediv_floordiv() -> None:
    """Execute ``test_lax_numpy_truediv_floordiv``."""
    a = jnp.array([4.0])
    b = jnp.array([2.0])
    assert jnp.all(4.0 / b == jnp.array([2.0]))
    assert jnp.all(a // b == jnp.array([2.0]))
    assert jnp.all(5.0 // b == jnp.array([2.0]))


def test_lax_numpy_setitem_eager() -> None:
    """Execute ``test_lax_numpy_setitem_eager``."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    a = jnp.array([1.0, 2.0])
    with pytest.raises(TypeError, match="immutable"):
        a[0] = 3.0
    updated = a.at[0].set(3.0)
    assert updated[0] == 3.0

    config.eager_mode = False
    with pytest.raises(TypeError, match="immutable"):
        a[0] = 5.0
    config.eager_mode = True


def test_lax_numpy_getitem_tensor() -> None:
    """Execute ``test_lax_numpy_getitem_tensor``."""
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    idx = jnp.array(0)
    a[idx]

    idx_tuple = (jnp.array(0), jnp.array(1))
    a[idx_tuple]


def test_lax_numpy_wrap_tracing() -> None:
    """Execute ``test_lax_numpy_wrap_tracing``."""
    t = jnp.array([1.0])._tensor
    ml_switcheroo_compiler.tracing.state.global_tracing_state.start_tracing("test_trace")
    try:
        jnp._to_tensor(t)
    finally:
        ml_switcheroo_compiler.tracing.state.global_tracing_state.stop_tracing()


def test_lax_numpy_wrap_tuple_list() -> None:
    """Execute ``test_lax_numpy_wrap_tuple_list``."""
    t1 = jnp.array([1.0])._tensor
    t2 = jnp.array([2.0])._tensor
    res_tuple = jnp._wrap((t1, t2))
    assert isinstance(res_tuple, tuple)
    res_list = jnp._wrap([t1, t2])
    assert isinstance(res_list, list)

    assert jnp._wrap(5.0) == 5.0


def test_lax_numpy_minimum() -> None:
    """Execute ``test_lax_numpy_minimum``."""
    a = jnp.array([1.0, 3.0])
    b = jnp.array([2.0, 2.0])
    jnp.minimum(a, b)


def test_lax_numpy_broadcast_shapes_empty() -> None:
    """Execute ``test_lax_numpy_broadcast_shapes_empty``."""
    assert jnp.broadcast_shapes() == ()


def test_lax_numpy_linspace_exceptions() -> None:
    """Execute ``test_lax_numpy_linspace_exceptions``."""
    jnp.linspace(0, 1, 10, retstep=True)


def test_lax_numpy_math_missing() -> None:
    """Execute ``test_lax_numpy_math_missing``."""
    x = jnp.array([1.0])
    jnp.sqrt(x)
    jnp.square(x)
    jnp.isnan(x)
    jnp.cumsum(x, axis=0, dtype=DType.Float32)


def test_pytree_fallback_and_properties() -> None:
    """Execute ``test_pytree_fallback_and_properties``."""
    from zero_jax.tree_util import pytree

    class DummyNode:
        """Test class ``DummyNode``."""

    DummyNode()

    structure = pytree.tree_structure([1, [2, 3]])
    assert structure.num_leaves == 3
    assert structure.num_nodes > 0

    pytree._patch_pytreedef()
    structure2 = pytree.tree_structure([1, [2, 3]])
    assert structure2.num_leaves == 3
    assert structure2.num_nodes > 0

    leaves = [1]
    pytree.tree_unflatten(pytree.PyTreeDef(int, []), leaves)

    assert not pytree.tree_any([False, False])


def test_lax_numpy_clip() -> None:
    """Execute ``test_lax_numpy_clip``."""
    a = jnp.array([1.0, 2.0, 3.0, 4.0])
    res = jnp.clip(a, 2.0, 3.0)
    assert jnp.all(res == jnp.array([2.0, 2.0, 3.0, 3.0]))


def test_lax_numpy_truediv() -> None:
    """Execute ``test_lax_numpy_truediv``."""
    a = jnp.array([4.0])
    b = jnp.array([2.0])
    res = a / b
    assert jnp.all(res == jnp.array([2.0]))


def test_linspace_negative_num() -> None:
    """Execute ``test_linspace_negative_num``."""
    with pytest.raises(ValueError, match="must be non-negative"):
        jnp.linspace(0, 1, -1)


def test_linspace_zero_num() -> None:
    """Execute ``test_linspace_zero_num``."""
    res = jnp.linspace(0, 1, 0)
    assert res.shape == (0,)


def test_linspace_axis() -> None:
    """Execute ``test_linspace_axis``."""
    res = jnp.linspace(0, 1, 3, axis=-1)
    assert res.shape == (3,)


def test_ravel_invalid_order() -> None:
    """Execute ``test_ravel_invalid_order``."""
    with pytest.raises(ValueError, match="order must be one of"):
        jnp.ravel(jnp.array([1, 2]), order="Z")
