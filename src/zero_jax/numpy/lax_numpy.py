"""JAX-like numpy API backed by ml-switcheroo-compiler."""

from typing import Any, Tuple, List, Optional
import uuid
import numpy as np

from ml_switcheroo_compiler.tracing import _tracer, ProxyTensor
from ml_switcheroo_ir import LogicalNode


def _unary_op(x: Any, op_type: str) -> Any:
    """Helper for unary operations."""
    if not _tracer.is_tracing:
        # Fallback to standard numpy if not tracing
        if op_type == "Sin":
            return np.sin(x)
        if op_type == "Cos":
            return np.cos(x)
        if op_type == "Exp":
            return np.exp(x)
        if op_type == "Log":
            return np.log(x)
        if op_type == "Transpose":
            return np.transpose(x)
        raise NotImplementedError(f"Eager {op_type} not implemented")

    if not isinstance(x, ProxyTensor):
        x_id = str(uuid.uuid4())
        node = LogicalNode(
            id=x_id, op_type="Constant", attributes={"value": x}, shape_metadata=()
        )
        _tracer.add_node(node)
        x = ProxyTensor(id=x_id, shape=())

    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id, op_type=op_type, inputs=[x.id], shape_metadata=x.shape
    )
    _tracer.add_node(node)
    return ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype)


def sin(x: Any) -> Any:
    return _unary_op(x, "Sin")


def cos(x: Any) -> Any:
    return _unary_op(x, "Cos")


def exp(x: Any) -> Any:
    return _unary_op(x, "Exp")


def log(x: Any) -> Any:
    return _unary_op(x, "Log")


def transpose(x: Any, axes: Optional[List[int]] = None) -> Any:
    if not _tracer.is_tracing:
        return np.transpose(x, axes=axes)

    if not isinstance(x, ProxyTensor):
        x_id = str(uuid.uuid4())
        node = LogicalNode(
            id=x_id, op_type="Constant", attributes={"value": x}, shape_metadata=()
        )
        _tracer.add_node(node)
        x = ProxyTensor(id=x_id, shape=())

    out_shape = tuple(x.shape[i] for i in axes) if axes else tuple(reversed(x.shape))
    out_id = str(uuid.uuid4())
    attrs = {"perm": axes} if axes else {}
    node = LogicalNode(
        id=out_id,
        op_type="Transpose",
        inputs=[x.id],
        attributes=attrs,
        shape_metadata=out_shape,
    )
    _tracer.add_node(node)
    return ProxyTensor(id=out_id, shape=out_shape, dtype=x.dtype)


def reshape(x: Any, newshape: Tuple[int, ...]) -> Any:
    if not _tracer.is_tracing:
        return np.reshape(x, newshape)

    if not isinstance(x, ProxyTensor):
        x_id = str(uuid.uuid4())
        node = LogicalNode(
            id=x_id, op_type="Constant", attributes={"value": x}, shape_metadata=()
        )
        _tracer.add_node(node)
        x = ProxyTensor(id=x_id, shape=())

    # Create a shape node
    shape_id = str(uuid.uuid4())
    shape_node = LogicalNode(
        id=shape_id,
        op_type="Constant",
        attributes={"value": list(newshape)},
        shape_metadata=(len(newshape),),
    )
    _tracer.add_node(shape_node)

    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Reshape",
        inputs=[x.id, shape_id],
        shape_metadata=tuple(newshape),
    )
    _tracer.add_node(node)
    return ProxyTensor(id=out_id, shape=tuple(newshape), dtype=x.dtype)


def broadcast_to(x: Any, shape: Tuple[int, ...]) -> Any:
    if not _tracer.is_tracing:
        return np.broadcast_to(x, shape)

    if not isinstance(x, ProxyTensor):
        x_id = str(uuid.uuid4())
        node = LogicalNode(
            id=x_id, op_type="Constant", attributes={"value": x}, shape_metadata=()
        )
        _tracer.add_node(node)
        x = ProxyTensor(id=x_id, shape=())

    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Expand",
        inputs=[x.id],
        attributes={"shape": list(shape)},
        shape_metadata=tuple(shape),
    )
    _tracer.add_node(node)
    return ProxyTensor(id=out_id, shape=tuple(shape), dtype=x.dtype)


def concatenate(arrays: List[Any], axis: int = 0) -> Any:
    if not _tracer.is_tracing:
        return np.concatenate(arrays, axis=axis)

    inputs = []
    shapes = []
    dtype = "float32"
    for arr in arrays:
        if not isinstance(arr, ProxyTensor):
            arr_id = str(uuid.uuid4())
            node = LogicalNode(
                id=arr_id,
                op_type="Constant",
                attributes={"value": arr},
                shape_metadata=(),
            )
            _tracer.add_node(node)
            inputs.append(arr_id)
            shapes.append(getattr(arr, "shape", ()))
        else:
            inputs.append(arr.id)
            shapes.append(arr.shape)
            dtype = arr.dtype

    # Compute output shape
    out_shape = list(shapes[0])
    for s in shapes[1:]:
        out_shape[axis] += s[axis]

    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Concat",
        inputs=inputs,
        attributes={"axis": axis},
        shape_metadata=tuple(out_shape),
    )
    _tracer.add_node(node)
    return ProxyTensor(id=out_id, shape=tuple(out_shape), dtype=dtype)


def where(condition: Any, x: Any, y: Any) -> Any:
    if not _tracer.is_tracing:
        return np.where(condition, x, y)

    inputs = []
    shapes = []
    dtype = "float32"
    for item in [condition, x, y]:
        if not isinstance(item, ProxyTensor):
            item_id = str(uuid.uuid4())
            node = LogicalNode(
                id=item_id,
                op_type="Constant",
                attributes={"value": item},
                shape_metadata=(),
            )
            _tracer.add_node(node)
            inputs.append(item_id)
            shapes.append(getattr(item, "shape", ()))
        else:
            inputs.append(item.id)
            shapes.append(item.shape)
            if item is not condition:
                dtype = item.dtype

    # Simplified broadcast shape
    out_shape = shapes[1]
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id, op_type="Where", inputs=inputs, shape_metadata=out_shape
    )
    _tracer.add_node(node)
    return ProxyTensor(id=out_id, shape=out_shape, dtype=dtype)


def einsum(subscripts: str, *operands: Any) -> Any:
    if not _tracer.is_tracing:
        return np.einsum(subscripts, *operands)

    inputs = []
    for item in operands:
        if not isinstance(item, ProxyTensor):
            item_id = str(uuid.uuid4())
            node = LogicalNode(
                id=item_id,
                op_type="Constant",
                attributes={"value": item},
                shape_metadata=(),
            )
            _tracer.add_node(node)
            inputs.append(item_id)
        else:
            inputs.append(item.id)

    # Minimal shape inference (mocked for now as empty tuple)
    out_shape = ()
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Einsum",
        inputs=inputs,
        attributes={"equation": subscripts},
        shape_metadata=out_shape,
    )
    _tracer.add_node(node)
    return ProxyTensor(id=out_id, shape=out_shape, dtype="float32")


# Common math aliases that use proxy tensors natively
def add(x: Any, y: Any) -> Any:
    if isinstance(x, ProxyTensor):
        return x + y
    if isinstance(y, ProxyTensor):
        return y + x
    return np.add(x, y)


def multiply(x: Any, y: Any) -> Any:
    if isinstance(x, ProxyTensor):
        return x * y
    if isinstance(y, ProxyTensor):
        return y * x
    return np.multiply(x, y)

def maximum(x: Any, y: Any) -> Any:
    return np.maximum(x, y)

def max(x: Any, axis: Any = None, keepdims: bool = False, where: Any = None, initial: Any = None) -> Any:
    kwargs = {}
    if where is not None: kwargs['where'] = where
    if initial is not None: kwargs['initial'] = initial
    return np.max(x, axis=axis, keepdims=keepdims, **kwargs)

def sum(x: Any, axis: Any = None, keepdims: bool = False, where: Any = None) -> Any:
    kwargs = {}
    if where is not None: kwargs['where'] = where
    return np.sum(x, axis=axis, keepdims=keepdims, **kwargs)

def zeros_like(x: Any, dtype: Any = None) -> Any:
    return np.zeros_like(x, dtype=dtype)

def zeros(shape: Any, dtype: Any = None) -> Any:
    return np.zeros(shape, dtype=dtype)

def abs(x: Any) -> Any:
    return np.abs(x)

def mean(x: Any, axis: Any = None, keepdims: bool = False) -> Any:
    return np.mean(x, axis=axis, keepdims=keepdims)

inf = np.inf

def array(x: Any, dtype: Any = None) -> Any:
    return np.array(x, dtype=dtype)

def isfinite(x):
    import numpy as onp
    return onp.isfinite(x)

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    import numpy as onp
    return onp.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

def array_equal(a1, a2, equal_nan=False):
    import numpy as onp
    return onp.array_equal(a1, a2, equal_nan=equal_nan)

def broadcast_shapes(*shapes):
    import numpy as onp
    return onp.broadcast_shapes(*shapes)
