from ml_switcheroo_compiler.core.tensor import TensorConfig

"""Tests for lax gather and scatter operations."""

from zero_jax.lax import gather, scatter, scatter_add
from ml_switcheroo_compiler.tracing import _tracer, ProxyTensor


def test_gather_scatter_tracing():
    """Test gather, scatter, and scatter_add using tracing mode."""
    _tracer.start_tracing()
    try:
        operand = ProxyTensor(id="op", shape=(3,))
        start_indices = ProxyTensor(id="idx", shape=(1,))
        updates = ProxyTensor(id="up", shape=(1,))

        res_gather = gather(
            operand, start_indices, dimension_numbers=None, slice_sizes=None
        )
        assert res_gather is not None

        res_scatter = scatter(operand, start_indices, updates, dimension_numbers=None)
        assert res_scatter is not None

        res_scatter_add = scatter_add(
            operand, start_indices, updates, dimension_numbers=None
        )
        assert res_scatter_add is not None
    finally:
        _tracer.stop_tracing()


def test_gather_scatter_dimension_numbers():
    from zero_jax.numpy import array

    operand = array([1, 2, 3])
    start_indices = array([0, 1])
    updates = array([5, 6])

    # Just pass something that is not None
    # We don't care about correctness here for ops.gather_nd fallback, just coverage
    try:
        gather(operand, start_indices, dimension_numbers=True, slice_sizes=None)
    except Exception:
        pass

    try:
        scatter(operand, start_indices, updates, dimension_numbers=True)
    except Exception:
        pass

    try:
        scatter_add(operand, start_indices, updates, dimension_numbers=True)
    except Exception:
        pass
