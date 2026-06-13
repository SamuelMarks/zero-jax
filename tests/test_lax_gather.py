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
