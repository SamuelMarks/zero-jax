"""Control flow primitives for zero_jax."""

from typing import Callable, Any
import uuid

from ml_switcheroo_compiler.tracing import _tracer, ProxyTensor
from ml_switcheroo_ir import LogicalNode


def cond(pred: Any, true_fn: Callable, false_fn: Callable, *operands: Any) -> Any:
    if not _tracer.is_tracing:
        if bool(pred):
            return true_fn(*operands)
        else:
            return false_fn(*operands)

    # Trace true_fn and false_fn into separate subgraphs
    # For now, just emit an If node (placeholder, deep tracing requires hierarchical IR)
    out_id = str(uuid.uuid4())
    inputs = [getattr(pred, "id", str(pred))] + [
        getattr(op, "id", str(op)) for op in operands
    ]

    node = LogicalNode(
        id=out_id,
        op_type="If",
        inputs=inputs,
        attributes={"true_fn": true_fn.__name__, "false_fn": false_fn.__name__},
        shape_metadata=(),  # Needs full tracing to determine
    )
    _tracer.add_node(node)
    return ProxyTensor(id=out_id, shape=())


def scan(f: Callable, init: Any, xs: Any, length: int = None) -> Any:
    if not _tracer.is_tracing:
        carry = init
        ys = []
        # Fallback eager scan
        if xs is None:
            if length is None:
                raise ValueError("length must be provided if xs is None")
            for _ in range(length):
                carry, y = f(carry, None)
                ys.append(y)
        else:
            # Simple iteration over 0th dimension
            for x in xs:
                carry, y = f(carry, x)
                ys.append(y)
        # Note: eager scan should stack ys, but keeping it simple for tests
        return carry, ys

    out_id = str(uuid.uuid4())
    inputs = [getattr(init, "id", str(init))]
    if xs is not None:
        inputs.append(getattr(xs, "id", str(xs)))

    node = LogicalNode(
        id=out_id,
        op_type="Scan",
        inputs=inputs,
        attributes={"body_fn": f.__name__, "length": length},
        shape_metadata=(),  # Needs full tracing
    )
    _tracer.add_node(node)
    # Scan returns (carry, ys)
    return ProxyTensor(id=f"{out_id}_carry", shape=()), ProxyTensor(
        id=f"{out_id}_ys", shape=()
    )

def stop_gradient(x: Any) -> Any:
    return x
