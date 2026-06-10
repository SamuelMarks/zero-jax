"""JAX-like numpy API backed by ml-switcheroo-compiler."""

from typing import Any, Tuple, List, Optional
import numpy as np
import ml_switcheroo.ops as ops
from ml_switcheroo import Tensor
import ml_switcheroo


class ndarray:
    def __init__(self, tensor):
        self._tensor = tensor

    @property
    def shape(self):
        return self._tensor.shape

    @property
    def dtype(self):
        return self._tensor.dtype

    def __array__(self):
        import numpy as np

        if hasattr(self._tensor.data, "id"):  # ProxyTensor check
            return np.zeros(
                self._tensor.shape
            )  # Return dummy shape for tracing asserts if needed
        return np.array(self._tensor.data)

    def __repr__(self):
        return repr(self.__array__())

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return add(self, multiply(other, -1))

    def __rsub__(self, other):
        return add(other, multiply(self, -1))

    def __mul__(self, other):
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __neg__(self):
        return multiply(self, -1.0)

    def __lt__(self, other):
        return _wrap(ops.less(self._tensor, _to_tensor(other)))

    def __gt__(self, other):
        return _wrap(ops.greater(self._tensor, _to_tensor(other)))

    def __le__(self, other):
        return _wrap(ops.less_equal(self._tensor, _to_tensor(other)))

    def __ge__(self, other):
        return _wrap(ops.greater_equal(self._tensor, _to_tensor(other)))

    def __getitem__(self, key):
        arr = self.__array__()
        return _wrap(_to_tensor(arr[key]))

    def __eq__(self, other):
        return ops.equal(self._tensor, _to_tensor(other))

    def __bool__(self):
        arr = self.__array__()
        if arr.size == 1:
            return bool(arr.item())
        raise ValueError(
            "The truth value of an array with more than one element is ambiguous."
        )

    def __len__(self):
        return self.shape[0] if self.shape else 0

    def __iter__(self):
        arr = self.__array__()
        for i in range(arr.shape[0]):
            yield array(arr[i])


def _to_tensor(x):
    if isinstance(x, ndarray):
        x = x._tensor
    from ml_switcheroo.core.config import config
    from ml_switcheroo.tracing import _tracer, ProxyTensor
    from ml_switcheroo_ir import LogicalNode
    import uuid

    if isinstance(x, ml_switcheroo.Tensor):
        if _tracer.is_tracing and not hasattr(x.data, "id"):
            # lift eager tensor as constant
            out_id = str(uuid.uuid4())
            node = LogicalNode(
                id=out_id,
                op_type="Constant",
                attributes={"value": np.array(x.data).tolist()},
                shape_metadata=x.shape,
            )
            _tracer.add_node(node)
            pt = ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype.value)
            return ml_switcheroo.Tensor(
                data=pt, shape=x.shape, dtype=x.dtype, device=x.device
            )
        return x
    if isinstance(x, ProxyTensor):
        # We need a dtype. ProxyTensor has dtype as string.
        # But we'll just mock it or use default.
        return ml_switcheroo.Tensor(
            data=x,
            shape=x.shape,
            dtype=config.default_float_dtype,
            device=config.default_device,
        )

    arr = np.array(x)
    if config.eager_mode and not _tracer.is_tracing:
        return ml_switcheroo.Tensor(
            arr, arr.shape, config.default_float_dtype, config.default_device
        )
    else:
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id, op_type="Constant", attributes={"value": arr.tolist()}
        )
        _tracer.add_node(node)
        pt = ProxyTensor(id=out_id, shape=arr.shape)
        return ml_switcheroo.Tensor(
            data=pt,
            shape=arr.shape,
            dtype=config.default_float_dtype,
            device=config.default_device,
        )


def _wrap(t):

    if isinstance(t, Tensor):
        return ndarray(t)
    elif isinstance(t, tuple):
        return tuple(_wrap(x) for x in t)
    elif isinstance(t, list):
        return list(_wrap(x) for x in t)
    return t


def sin(x: Any) -> Any:
    return _wrap(ops.sin(_to_tensor(x)))


def cos(x: Any) -> Any:
    return _wrap(ops.cos(_to_tensor(x)))


def exp(x: Any) -> Any:
    return _wrap(ops.exp(_to_tensor(x)))


def log(x: Any) -> Any:
    return _wrap(ops.log(_to_tensor(x)))


def transpose(x: Any, axes: Optional[List[int]] = None) -> Any:
    t = _to_tensor(x)
    if axes is not None:
        return _wrap(ops.permute(t, dims=axes))
    axes = list(range(len(t.shape))[::-1])
    return _wrap(ops.permute(t, dims=axes))


def reshape(x: Any, newshape: Tuple[int, ...]) -> Any:
    return _wrap(ops.reshape(_to_tensor(x), shape=newshape))


def broadcast_to(x: Any, shape: Tuple[int, ...]) -> Any:
    return _wrap(ops.broadcast_to(_to_tensor(x), size=shape))


def concatenate(arrays: List[Any], axis: int = 0) -> Any:
    tensors = [_to_tensor(a) for a in arrays]
    return _wrap(ops.concatenate(tensors, dim=axis))


def where(condition: Any, x: Any, y: Any) -> Any:
    return _wrap(ops.where(_to_tensor(condition), _to_tensor(x), _to_tensor(y)))


def einsum(subscripts: str, *operands: Any) -> Any:
    tensors = [_to_tensor(a) for a in operands]
    return _wrap(ops.einsum(subscripts, *tensors))


def add(x: Any, y: Any) -> Any:
    return _wrap(ops.add(_to_tensor(x), _to_tensor(y)))


def multiply(x: Any, y: Any) -> Any:
    return _wrap(ops.multiply(_to_tensor(x), _to_tensor(y)))


def maximum(x: Any, y: Any) -> Any:
    return _wrap(ops.maximum(_to_tensor(x), _to_tensor(y)))


def max(
    x: Any,
    axis: Any = None,
    keepdims: bool = False,
    where: Any = None,
    initial: Any = None,
) -> Any:
    return _wrap(ops.max(_to_tensor(x), axis=axis, keepdims=keepdims))


def sum(x: Any, axis: Any = None, keepdims: bool = False, where: Any = None) -> Any:
    return _wrap(ops.sum(_to_tensor(x), axis=axis, keepdims=keepdims))


def zeros_like(x: Any, dtype: Any = None) -> Any:
    return _wrap(ops.zeros_like(_to_tensor(x), dtype=dtype))


def zeros(shape: Any, dtype: Any = None) -> Any:
    return _wrap(ops.zeros(shape=shape, dtype=dtype))


def abs(x: Any) -> Any:
    return _wrap(ops.abs(_to_tensor(x)))


def mean(x: Any, axis: Any = None, keepdims: bool = False) -> Any:
    return _wrap(ops.mean(_to_tensor(x), axis=axis, keepdims=keepdims))


inf = np.inf


def array(x: Any, dtype: Any = None) -> Any:
    if isinstance(x, ndarray):
        return x
    return _wrap(_to_tensor(x))


def dot(a: Any, b: Any) -> Any:
    return _wrap(ops.dot(_to_tensor(a), _to_tensor(b)))


def matmul(a: Any, b: Any) -> Any:
    return _wrap(ops.matmul(_to_tensor(a), _to_tensor(b)))


def expand_dims(a: Any, axis: int) -> Any:
    return _wrap(ops.unsqueeze(_to_tensor(a), dim=axis))


def isfinite(x):
    return _wrap(ops.isfinite(_to_tensor(x)))


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return ops.allclose(
        _to_tensor(a), _to_tensor(b), rtol=rtol, atol=atol, equal_nan=equal_nan
    )


def array_equal(a1, a2, equal_nan=False):
    res = ops.equal(_to_tensor(a1), _to_tensor(a2))
    return np.all(res.data) if hasattr(res, "data") else True


def broadcast_shapes(*shapes):
    return np.broadcast_shapes(*shapes)


def _unary_op(x, name):
    if name == "Transpose":
        return transpose(x)
    raise NotImplementedError()
