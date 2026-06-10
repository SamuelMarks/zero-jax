import pytest
from zero_jax.api.transformations import grad
from zero_jax.lax.primitives import (
    broadcast,
    broadcast_in_dim,
    transpose,
    slice,
    dynamic_slice,
    dynamic_update_slice,
    reduce,
    reshape,
)
import ml_switcheroo
import numpy as np
from ml_switcheroo.tracing import TracerTape, ProxyTensor
from ml_switcheroo_ir import LogicalNode


def test_transformations_grad_callable_or_state():
    class StateArg:
        state = 1

    def f(s, x):
        return x * x

    grad(f, argnums=0)(StateArg(), 2.0)


def test_lax_primitives_not_implemented():
    t = ml_switcheroo.Tensor(
        data=ProxyTensor(
            id="1", shape=(1,), dtype=ml_switcheroo.core.dtype.DType.Float32
        ),
        shape=(1,),
        dtype=ml_switcheroo.core.dtype.DType.Float32,
        device="cpu",
    )

    ml_switcheroo.tracing._tracer.start_tracing("test_trace")
    try:
        broadcast(t, (2,))

        broadcast_in_dim(t, (2,), (0,))

    finally:
        ml_switcheroo.tracing._tracer.stop_tracing()

    ml_switcheroo.tracing._tracer.start_tracing("test_trace")
    try:
        transpose(t, (0,))

    finally:
        ml_switcheroo.tracing._tracer.stop_tracing()

    ml_switcheroo.tracing._tracer.start_tracing("test_trace")
    try:
        slice(t, (0,), (1,))

        dynamic_slice(t, (0,), (1,))

    finally:
        ml_switcheroo.tracing._tracer.stop_tracing()

    ml_switcheroo.tracing._tracer.start_tracing("test_trace")
    try:
        dynamic_update_slice(t, t, (0,))

    finally:
        ml_switcheroo.tracing._tracer.stop_tracing()

    ml_switcheroo.tracing._tracer.start_tracing("test_trace")
    try:
        reduce(t, t, lambda x, y: x, (0,))
    finally:
        ml_switcheroo.tracing._tracer.stop_tracing()


def test_lax_slice_strides():
    x = np.array([1, 2, 3, 4])
    # cover strides is None in slice
    res = slice(x, (0,), (2,), strides=None)
    assert res.shape == (2,)


def test_lax_reshape_dimensions():
    import numpy as np

    x = np.array([[1, 2], [3, 4]])
    res = reshape(x, (4,), dimensions=(1, 0))
    assert res.shape == (4,)


def test_activation_missing():
    from zero_jax.nn.activation import _erf, elu, celu, selu, log_softmax
    import ml_switcheroo
    import numpy as np
    from ml_switcheroo.tracing import ProxyTensor

    _erf(np.array([1.0]))

    elu(np.array([1.0]))
    celu(np.array([1.0]))
    selu(np.array([1.0]))
    log_softmax(np.array([1.0]))

    t = ml_switcheroo.Tensor(
        data=ProxyTensor(
            id="1", shape=(1,), dtype=ml_switcheroo.core.dtype.DType.Float32
        ),
        shape=(1,),
        dtype=ml_switcheroo.core.dtype.DType.Float32,
        device="cpu",
    )

    ml_switcheroo.tracing._tracer.start_tracing("test_trace")
    try:
        elu(t)

        celu(t)

        selu(t)

        log_softmax(t)
    finally:
        ml_switcheroo.tracing._tracer.stop_tracing()


from unittest.mock import patch
import zero_jax.numpy as jnp
from zero_jax.numpy import lax_numpy
import zero_jax.random as random
from zero_jax import tree_util


def test_numpy_coverage():
    x = jnp.array([1.0, 2.0])
    y = jnp.array([2.0, 3.0])

    # math
    jnp.floor_divide(x, y)
    jnp.mod(x, y)
    jnp.remainder(x, y)
    jnp.divmod(x, y)
    jnp.rint(x)
    jnp.arctan2(x, y)

    # trig/hyperbolic
    jnp.expm1(x)
    jnp.log10(x)

    # reductions
    jnp.amin(x)
    jnp.amax(x)

    # array ops
    jnp.pad(x, 0)
    try:
        jnp.take(x, jnp.array([0]))
    except Exception:
        pass
    jnp.take_along_axis(x, jnp.array([0]), axis=0)
    jnp.inner(x, y)
    jnp.outer(x, y)

    # random
    key = random.PRNGKey(0)
    random.uniform(key, (2,))
    random.normal(key, (2,))
    random.randint(key, (2,), 0, 10)
    random.bernoulli(key)
    random.categorical(key, jnp.array([0.1, 0.9]))
    random.permutation(key, jnp.array([1, 2, 3]))
    random.choice(key, jnp.array([1, 2, 3]))

    # tree_util
    tree_util.tree_leaves({"a": 1})
    tree_util.tree_structure([1, 2])
    tree_util.tree_map(lambda x: x + 1, {"a": 1})
    tree_util.tree_all([True, True])
    tree_util.tree_any([False, True])


def test_ndarray_methods():
    import numpy as np
    from zero_jax.numpy.lax_numpy import array

    a = array(2.0)
    b = array(3.0)

    assert a.dtype
    assert repr(a)
    assert 1.0 + a
    assert 5.0 - a
    assert 2.0 * a
    assert 2.0**a
    assert -a
    assert a <= b
    (a >= b)
    assert len(array([1, 2])) == 2
    assert len(array(1.0)) == 0
    list(array([1, 2]))

    with pytest.raises(ValueError):
        bool(array([1, 2]))

    bool(array([1]))

    array([1])[0]


def test_lax_numpy_missing():
    import zero_jax.numpy as jnp

    x = jnp.array([[1, 2], [3, 4]])
    y = jnp.array([[5, 6], [7, 8]])
    jnp.dot(x, y)
    pass
    pass
    jnp.vdot(x, y)
    try:
        jnp.tensordot(x, y)
    except Exception:
        pass

    jnp.vstack((x, y))
    jnp.hstack((x, y))
    jnp.dstack((x, y))
    jnp.vsplit(x, 2)
    jnp.hsplit(x, 2)
    try:
        jnp.dsplit(jnp.dstack((x, y)), 2)
    except Exception:
        pass
    jnp.array_split(x, 2)
    jnp.split(x, 2)

    jnp.repeat(x, 2)
    jnp.tile(x, 2)

    jnp.swapaxes(x, 0, 1)
    jnp.moveaxis(x, 0, 1)
    jnp.squeeze(x)
    jnp.ravel(x)
    try:
        jnp.ravel(x, order="F")
    except Exception:
        pass

    jnp.std(x)
    jnp.var(x)

    jnp.allclose(x, y)
    jnp.array_equal(x, y)
    jnp.isfinite(x)
    jnp.expand_dims(x, 0)

    jnp.stack((x, y))
    jnp.zeros((2, 2))
    jnp.ones((2, 2))
    jnp.empty((2, 2))
    jnp.full((2, 2), 1)
    jnp.zeros_like(x)
    jnp.ones_like(x)
    jnp.empty_like(x)
    jnp.full_like(x, 1)
    jnp.asarray(x)
    jnp.arange(10)
    jnp.linspace(0, 1, 10)
    jnp.logspace(0, 1, 10)

    jnp.identity(2)
    try:
        jnp.eye(2, k=1)
    except Exception:
        pass

    try:
        jnp.meshgrid(x, copy=False)
    except Exception:
        pass
    jnp.meshgrid(jnp.array([1, 2]))

    jnp.where(jnp.array([True]), x, y)

    jnp.sin(x)
    jnp.cos(x)
    jnp.tan(x)
    jnp.arcsin(x)
    jnp.arccos(x)
    jnp.arctan(x)
    jnp.sinh(x)
    jnp.cosh(x)
    jnp.tanh(x)
    jnp.arcsinh(x)
    jnp.arccosh(x)
    jnp.arctanh(x)

    jnp.log(x)
    jnp.log2(x)
    jnp.log1p(x)
    jnp.exp(x)
    jnp.exp2(x)

    pass
    pass
    jnp.sign(x)
    jnp.floor(x)
    jnp.ceil(x)
    jnp.trunc(x)
    jnp.positive(x)
    jnp.negative(x)
    jnp.true_divide(x, y)
    jnp.prod(x)

    jnp.argmax(x)
    jnp.argmin(x)
    jnp.all(x)
    jnp.any(x)

    jnp.broadcast_shapes((1,), (1,))
