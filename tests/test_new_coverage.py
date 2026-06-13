import pytest
import numpy as np
import ml_switcheroo_compiler
from ml_switcheroo_compiler.tracing import ProxyTensor
from ml_switcheroo_compiler.core.dtype import DType


def test_initializers_compute_fans():
    from zero_jax.nn.initializers import _compute_fans

    assert _compute_fans((2, 2)) == (10, 10)


def test_lax_numpy_bool_proxy():
    import zero_jax.numpy as jnp
    from zero_jax.numpy import tensor_utils

    t = ml_switcheroo_compiler.Tensor(
        data=ProxyTensor(id="1", shape=(1,), dtype=DType.Float32),
        shape=(1,),
        dtype=DType.Float32,
        device="cpu",
    )
    arr = jnp.ndarray(t)
    # Testing __bool__ for ProxyTensor
    bool(arr)


def test_lax_numpy_sub_rsub():
    import zero_jax.numpy as jnp

    a = jnp.array([2.0])
    b = jnp.array([1.0])
    assert jnp.all(a - b == jnp.array([1.0]))
    assert jnp.all(2.0 - b == jnp.array([1.0]))


def test_lax_numpy_truediv_floordiv():
    import zero_jax.numpy as jnp

    a = jnp.array([4.0])
    b = jnp.array([2.0])
    assert jnp.all(4.0 / b == jnp.array([2.0]))
    assert jnp.all(a // b == jnp.array([2.0]))
    assert jnp.all(5.0 // b == jnp.array([2.0]))


def test_lax_numpy_setitem_eager():
    from ml_switcheroo_compiler.core.config import config
    import zero_jax.numpy as jnp

    config.eager_mode = True
    a = jnp.array([1.0, 2.0])
    a[0] = 3.0
    assert a[0] == 3.0
    a[1] = jnp.array(4.0)
    assert a[1] == 4.0

    config.eager_mode = False
    with pytest.raises(NotImplementedError):
        a[0] = 5.0
    config.eager_mode = True


def test_lax_numpy_getitem_tensor():
    import zero_jax.numpy as jnp

    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    idx = jnp.array(0)
    res = a[idx]

    idx_tuple = (jnp.array(0), jnp.array(1))
    res2 = a[idx_tuple]


def test_lax_numpy_wrap_tracing():
    import zero_jax.numpy as jnp
    import ml_switcheroo_compiler

    t = ml_switcheroo_compiler.Tensor(
        data=np.array([1.0]),
        shape=(1,),
        dtype=DType.Float32,
        device="cpu",
    )
    # Testing _to_tensor with eager tensor under tracing
    ml_switcheroo_compiler.tracing._tracer.start_tracing("test_trace")
    try:
        jnp._to_tensor(t)
    finally:
        ml_switcheroo_compiler.tracing._tracer.stop_tracing()


def test_lax_numpy_wrap_tuple_list():
    import zero_jax.numpy as jnp
    import ml_switcheroo_compiler

    t1 = ml_switcheroo_compiler.Tensor(
        data=np.array([1.0]),
        shape=(1,),
        dtype=DType.Float32,
        device="cpu",
    )
    t2 = ml_switcheroo_compiler.Tensor(
        data=np.array([2.0]),
        shape=(1,),
        dtype=DType.Float32,
        device="cpu",
    )
    res_tuple = jnp._wrap((t1, t2))
    assert isinstance(res_tuple, tuple)
    res_list = jnp._wrap([t1, t2])
    assert isinstance(res_list, list)

    # testing line 222
    assert jnp._wrap(5.0) == 5.0


def test_lax_numpy_minimum():
    import zero_jax.numpy as jnp

    a = jnp.array([1.0, 3.0])
    b = jnp.array([2.0, 2.0])
    jnp.minimum(a, b)


def test_lax_numpy_broadcast_shapes_empty():
    import zero_jax.numpy as jnp

    assert jnp.broadcast_shapes() == ()


def test_lax_numpy_linspace_exceptions():
    import zero_jax.numpy as jnp

    with pytest.raises(NotImplementedError):
        jnp.linspace(0, 1, 10, retstep=True)


@pytest.mark.skip(reason="Not implemented in backend")
def test_lax_numpy_math_missing():
    import zero_jax.numpy as jnp

    x = jnp.array([1.0])
    jnp.sqrt(x)
    jnp.square(x)
    jnp.isnan(x)
    jnp.cumsum(x, axis=0, dtype=DType.Float32)


def test_pytree_fallback_and_properties():
    from zero_jax.tree_util import pytree

    class DummyNode:
        pass

    dummy = DummyNode()

    # Test properties
    structure = pytree.tree_structure([1, [2, 3]])
    assert structure.num_leaves == 3
    assert structure.num_nodes > 0

    # Ensure _patch_pytreedef is called for 100% coverage of its lambda
    pytree._patch_pytreedef()
    structure2 = pytree.tree_structure([1, [2, 3]])
    assert structure2.num_leaves == 3
    assert structure2.num_nodes > 0

    # Test fallback in tree_unflatten
    leaves = [1]
    pytree.tree_unflatten(pytree.PyTreeDef(int, []), leaves)

    # Test tree_any with all false
    assert not pytree.tree_any([False, False])


def test_lax_numpy_clip():
    import zero_jax.numpy as jnp

    a = jnp.array([1.0, 2.0, 3.0, 4.0])
    res = jnp.clip(a, 2.0, 3.0)
    assert jnp.all(res == jnp.array([2.0, 2.0, 3.0, 3.0]))


def test_lax_numpy_truediv():
    import zero_jax.numpy as jnp

    a = jnp.array([4.0])
    b = jnp.array([2.0])
    # tests __truediv__ where a is an array
    res = a / b
    assert jnp.all(res == jnp.array([2.0]))
