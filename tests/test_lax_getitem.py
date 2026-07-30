import pytest
from ml_switcheroo_compiler.core.config import EagerMode
from ml_switcheroo_compiler.core.tensor import TensorConfig
from ml_switcheroo_ir import LogicalNode

from zero_jax.numpy import array, array_equal


def test_getitem_eager():
    with EagerMode():
        x = array([1, 2, 3])
        assert bool(x[0] == 1)
        assert bool(x[1] == 2)

        y = array([[1, 2], [3, 4]])
        assert bool(y[0, 1] == 2)
        assert bool(y[array(1), array(1)] == 4)


def test_getitem_tracing():
    import uuid

    import ml_switcheroo_compiler as compiler
    from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

    from zero_jax.numpy.lax_numpy import array, ndarray

    prev_graph = _tracer.active_graph
    is_tracing = _tracer.is_tracing
    graph = _tracer.start_tracing()
    try:
        in_id = str(uuid.uuid4())
        node = LogicalNode(id=in_id, op_type="Input", inputs=[], shape_metadata=(5, 5))
        graph.nodes[in_id] = node
        proxy = ProxyTensor(id=in_id, shape=(5, 5), dtype="float32")
        proxy_tensor = compiler.Tensor(
            data=proxy,
            config=TensorConfig(
                shape=(5, 5),
                dtype=compiler.core.dtype.DType.Float32,
                device=compiler.core.config.default_device,
            ),
        )
        x = ndarray(proxy_tensor)

        # Slices
        y1 = x[1:4]
        y2 = x[1:4, None, :]
        y3 = x[2]  # fallback GetItem
    finally:
        _tracer.stop_tracing()
        _tracer.active_graph = prev_graph
        _tracer.is_tracing = is_tracing


def test_getitem_tracing_padding():
    import uuid

    import ml_switcheroo_compiler as compiler
    from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
    from ml_switcheroo_ir import LogicalNode

    from zero_jax.numpy.lax_numpy import array, ndarray

    prev_graph = _tracer.active_graph
    is_tracing = _tracer.is_tracing
    graph = _tracer.start_tracing()
    try:
        in_id = str(uuid.uuid4())
        node = LogicalNode(id=in_id, op_type="Input", inputs=[], shape_metadata=(5, 5))
        graph.nodes[in_id] = node
        proxy = ProxyTensor(id=in_id, shape=(5, 5), dtype="float32")
        proxy_tensor = compiler.Tensor(
            data=proxy,
            config=TensorConfig(
                shape=(5, 5),
                dtype=compiler.core.dtype.DType.Float32,
                device=compiler.core.config.default_device,
            ),
        )
        x = ndarray(proxy_tensor)

        # 2D array, but we only slice 1st dim with a tuple containing one slice!
        y = x[(slice(1, 4),)]
    finally:
        _tracer.stop_tracing()
        _tracer.active_graph = prev_graph
        _tracer.is_tracing = is_tracing
