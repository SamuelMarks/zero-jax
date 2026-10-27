"""Control flow primitives for zero_jax."""

from __future__ import annotations

import uuid
from collections.abc import Callable

import ml_switcheroo_compiler.ops.control_flow as cf
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
from ml_switcheroo_ir import LogicalNode

from zero_jax._types import ArrayLikeSpec, PyTree
from zero_jax.numpy import stack
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap, ndarray
from zero_jax.tree_util import tree_flatten, tree_leaves, tree_map, tree_structure, tree_unflatten


class _CollectiveAxisFrame:
    """State frame tracking replica slices for an active ``vmap`` or ``pmap`` ``axis_name``."""

    def __init__(self, axis_name: str, axis_size: int) -> None:
        """Initialize ``_CollectiveAxisFrame``.

        Args:
            axis_name: Mapped axis identifier.
            axis_size: Number of slices along the mapped axis.
        """
        self.axis_name = axis_name
        self.axis_size = axis_size
        self.collecting = True
        self.call_idx = 0
        self.recorded_slices: dict[int, list[PyTree]] = {}


_ACTIVE_AXIS_FRAMES: dict[str, _CollectiveAxisFrame] = {}


def apply_axis_collective(op_kind: str, operand: PyTree, axis_name: PyTree) -> PyTree | None:
    """Evaluate a collective operation (``psum``, ``pmean``, ``all_gather``) across an active mapped axis.

    Args:
        op_kind: Collective kind (``"psum"``, ``"pmean"``, or ``"all_gather"``).
        operand: Per-replica operand array or PyTree.
        axis_name: Target axis name.

    Returns:
        PyTree | None: Collective result if ``axis_name`` is active, or ``None`` otherwise.
    """
    key = str(axis_name)
    if key not in _ACTIVE_AXIS_FRAMES:
        return None
    frame = _ACTIVE_AXIS_FRAMES[key]
    idx = frame.call_idx
    frame.call_idx += 1
    if frame.collecting:
        frame.recorded_slices.setdefault(idx, []).append(operand)
        return operand
    slices = frame.recorded_slices[idx]
    if op_kind == "psum":
        return tree_map(lambda *xs: stack(list(xs)).sum(axis=0), *slices)
    if op_kind == "pmean":
        return tree_map(lambda *xs: stack(list(xs)).mean(axis=0), *slices)
    return tree_map(lambda *xs: stack(list(xs)), *slices)


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


def scan(
    f: Callable,
    init: ArrayLikeSpec,
    xs: ArrayLikeSpec = None,
    length: int | None = None,
    reverse: bool = False,
    unroll: int = 1,
) -> ArrayLikeSpec:
    """Scan a function over leading array axes while carrying along state.

    Supports arbitrary PyTree ``init`` carries (such as ``(c, h)`` tuples,
    ``zero_flax.nnx.State``, and ``zero_optax`` states) and PyTree ``xs``
    sequence inputs, as well as ``reverse`` and ``unroll``.

    Args:
        f: The scanning function taking ``(carry, x)`` and returning ``(carry, y)``.
        init: Initial carry state PyTree.
        xs: Sequence or PyTree of sequences over which to scan.
        length: Optional scan length, required if ``xs`` is ``None``.
        reverse: If ``True``, run the scan in reverse order along the leading axis.
        unroll: Positive integer specifying how many scan iterations to unroll per chunk.

    Returns:
        ArrayLikeSpec: A tuple ``(final_carry, stacked_outputs)``.
    """
    if xs is None and length is None:
        raise ValueError("length must be provided if xs is None")

    if config.eager_mode and not _tracer.is_tracing:
        _, carry_treedef = tree_flatten(init)
        if xs is None:
            step_xs: list[PyTree] = [0] * int(length or 0)
        elif isinstance(xs, list) and all(not isinstance(item, ndarray) for item in xs):
            step_xs = list(xs)
        else:
            xs_leaves, xs_treedef = tree_flatten(xs)
            num_steps = int(length) if length is not None else len(xs_leaves[0])
            step_xs = [tree_unflatten(xs_treedef, [leaf[t] for leaf in xs_leaves]) for t in range(num_steps)]

        if reverse:
            step_xs.reverse()

        carry = init
        ys: list[PyTree] = []
        chunk = max(1, int(unroll))
        for start in range(0, len(step_xs), chunk):
            for x in step_xs[start : start + chunk]:
                next_carry, y = f(carry, x)
                carry = tree_unflatten(carry_treedef, tree_leaves(next_carry)) if carry_treedef.node_type is not type(None) else next_carry
                ys.append(y)

        if reverse:
            ys.reverse()

        if ys and any(isinstance(leaf, ndarray) for leaf in tree_leaves(ys[0])):
            y_leaves_0, y_treedef = tree_flatten(ys[0])
            all_y_leaves = [tree_flatten(y_step)[0] for y_step in ys]
            stacked_leaves = [stack([step_leaves[k] for step_leaves in all_y_leaves]) for k in range(len(y_leaves_0))]
            return carry, tree_unflatten(y_treedef, stacked_leaves)
        return carry, ys

    carry_t, ys_t = cf.scan(f, _to_tensor(init), _to_tensor(xs))
    return _wrap(carry_t), _wrap(ys_t)


_GRAPH_STOP_GRADIENT_SOURCES: dict[int, dict[str, str]] = {}


def get_stop_gradient_sources(graph_id: int) -> dict[str, str]:
    """Return and clear the stop-gradient source mapping for a trace graph.

    Args:
        graph_id: ``id(graph)`` of the active or completed trace graph.

    Returns:
        dict[str, str]: Mapping from detached node ID to source node ID.
    """
    return _GRAPH_STOP_GRADIENT_SOURCES.pop(graph_id, {})


def stop_gradient(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stop the flow of gradients during forward or reverse-mode differentiation.

    Args:
        x: The value or PyTree to stop gradients for.

    Returns:
        ArrayLikeSpec: The identical value, treated as a constant by gradient computations.
    """
    if _tracer.is_tracing and _tracer.active_graph is not None:
        active_graph = _tracer.active_graph
        graph_sources = _GRAPH_STOP_GRADIENT_SOURCES.setdefault(id(active_graph), {})

        def _detach_leaf(leaf: ArrayLikeSpec) -> ArrayLikeSpec:
            """Create a detached constant-like proxy node when ``leaf`` is being traced.

            Args:
                leaf: Input leaf.

            Returns:
                ArrayLikeSpec: Detached leaf.
            """
            t = _to_tensor(leaf)
            sg_id = str(uuid.uuid4())
            graph_sources[sg_id] = t.data.id
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


__all__ = [
    "_ACTIVE_AXIS_FRAMES",
    "_CollectiveAxisFrame",
    "apply_axis_collective",
    "cond",
    "get_stop_gradient_sources",
    "scan",
    "stop_gradient",
]
