"""Control flow primitives for zero_jax."""

from __future__ import annotations

from collections.abc import Callable

import ml_switcheroo_compiler.ops.control_flow as cf
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer

from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap


def cond(pred: ArrayLikeSpec, true_fn: Callable, false_fn: Callable, *operands: ArrayLikeSpec) -> ArrayLikeSpec:
    """Conditionally applies one of two functions.

    Args:
        pred: A boolean condition.
        true_fn: The function to execute if pred is true.
        false_fn: The function to execute if pred is false.
        *operands: Arguments to pass to the chosen function.

    Returns:
        The output of the conditionally executed function.
    """

    def wrapped_true() -> ArrayLikeSpec:
        """Executes the true branch.

        Returns:
            The tensor output of the true function.
        """
        return _to_tensor(true_fn(*operands))

    def wrapped_false() -> ArrayLikeSpec:
        """Executes the false branch.

        Returns:
            The tensor output of the false function.
        """
        return _to_tensor(false_fn(*operands))

    return _wrap(cf.cond(_to_tensor(pred), wrapped_true, wrapped_false))


def scan(f: Callable, init: ArrayLikeSpec, xs: ArrayLikeSpec, length: int | None = None) -> ArrayLikeSpec:
    """Scans a function over leading array axes while carrying along state.

    Args:
        f: The scanning function. Takes (carry, x) and returns (carry, y).
        init: The initial state.
        xs: The sequence of inputs over which to scan.
        length: Optional length of the scan, required if xs is None.

    Returns:
        A tuple of the final state and the stacked outputs.
    """
    if xs is None:
        if length is None:
            raise ValueError("length must be provided if xs is None")
        xs = [0] * length
    elif config.eager_mode:
        pass  # xs is iterable

    if config.eager_mode and not _tracer.is_tracing:
        carry = init
        ys = []
        for x in xs:
            carry, y = f(carry, x)
            ys.append(y)
        return carry, ys
    # tracing
    carry, ys = cf.scan(f, _to_tensor(init), _to_tensor(xs))
    return _wrap(carry), _wrap(ys)


_STOP_GRADIENT_SOURCES: dict[str, str] = {}


def stop_gradient(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stop the flow of gradients during forward or reverse-mode differentiation.

    Args:
        x: The value or PyTree to stop gradients for.

    Returns:
        ArrayLikeSpec: The identical value, treated as a constant by gradient computations.
    """
    if _tracer.is_tracing and _tracer.active_graph is not None:
        import uuid

        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
        from ml_switcheroo_ir import LogicalNode

        from zero_jax.numpy.lax_numpy import ndarray
        from zero_jax.tree_util import tree_map

        active_graph = _tracer.active_graph

        def _detach_leaf(leaf: ArrayLikeSpec) -> ArrayLikeSpec:
            """Create a detached constant-like proxy node when ``leaf`` is being traced.

            Args:
                leaf: Input leaf.

            Returns:
                ArrayLikeSpec: Detached leaf.
            """
            t = _to_tensor(leaf)
            sg_id = str(uuid.uuid4())
            _STOP_GRADIENT_SOURCES[sg_id] = t.data.id
            node = LogicalNode(
                id=sg_id,
                op_type="Input",
                inputs=[],
                shape_metadata=t.shape,
            )
            active_graph.nodes[sg_id] = node
            proxy = ProxyTensor(id=sg_id, shape=t.shape, dtype=t.dtype.value)
            return ndarray(Tensor(data=proxy, config=TensorConfig(shape=t.shape, dtype=t.dtype, device=t.device)))

        return tree_map(_detach_leaf, x)
    return x
