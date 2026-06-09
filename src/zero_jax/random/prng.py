"""PRNG state manipulation."""

from typing import Any
import numpy as np


def split(key: Any, num: int = 2) -> Any:
    """Splits a PRNG key into `num` new keys."""
    if isinstance(key, np.ndarray) or isinstance(key, list):
        # Mock deterministic eager split
        keys = []
        base = int(np.sum(key))
        for i in range(num):
            keys.append(np.array([base, i]))
        return np.array(keys) if num > 1 else keys[0]

    # Trace logic would emit a Split node
    from ml_switcheroo_compiler.tracing import _tracer, ProxyTensor
    from ml_switcheroo_ir import LogicalNode
    import uuid

    if _tracer.is_tracing and key is not None:
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="RandomSplit",
            inputs=[key.id],
            attributes={"num": num},
            shape_metadata=(num, 2),
        )
        _tracer.add_node(node)
        return ProxyTensor(id=out_id, shape=(num, 2))

    return None


def fold_in(key: Any, data: Any) -> Any:
    """Folds in data to a PRNG key."""
    if isinstance(key, np.ndarray) or isinstance(key, list):
        base = int(np.sum(key)) + int(np.sum(data))
        return np.array([base, 0])

    from ml_switcheroo_compiler.tracing import _tracer, ProxyTensor
    from ml_switcheroo_ir import LogicalNode
    import uuid

    if _tracer.is_tracing and key is not None and getattr(key, "id", None):
        out_id = str(uuid.uuid4())
        data_id = getattr(data, "id", None)
        inputs = [key.id]
        if data_id is not None:
            inputs.append(data_id)

        node = LogicalNode(
            id=out_id,
            op_type="RandomFoldIn",
            inputs=inputs,
            attributes={"data": data if data_id is None else None},
            shape_metadata=(2,),
        )
        _tracer.add_node(node)
        return ProxyTensor(id=out_id, shape=(2,))

    return None
