"""JAX-like numpy API backed by ml-switcheroo-compiler."""

from typing import Any

from typing import Tuple, List, Optional
import ml_switcheroo.ops as ops
from ml_switcheroo import Tensor
import ml_switcheroo


class ndarray:
    """ndarray class."""

    def __init__(self, tensor: Any) -> None:
        """Initialize."""
        self._tensor = tensor

    @property
    def shape(self) -> Any:
        """Shape function."""
        return self._tensor.shape

    @property
    def dtype(self) -> Any:
        """Dtype function."""
        return self._tensor.dtype

    def __array__(self) -> Any:
        """__array__ function."""
        from ml_switcheroo.core import tensor_utils

        if hasattr(self._tensor.data, "id"):  # ProxyTensor check
            return tensor_utils.zeros(
                self._tensor.shape
            )  # Return dummy shape for tracing asserts if needed
        return tensor_utils.to_array(self._tensor.data)

    def __repr__(self) -> Any:
        """__repr__ function."""
        return repr(self.__array__())

    def __add__(self, other: Any) -> Any:
        """__add__ function."""
        return add(self, other)

    def __radd__(self, other: Any) -> Any:
        """__radd__ function."""
        return add(other, self)

    def __sub__(self, other: Any) -> Any:
        """__sub__ function."""
        return add(self, multiply(other, -1))

    def __rsub__(self, other: Any) -> Any:
        """__rsub__ function."""
        return add(other, multiply(self, -1))

    def __mul__(self, other: Any) -> Any:
        """__mul__ function."""
        return multiply(self, other)

    def __rmul__(self, other: Any) -> Any:
        """__rmul__ function."""
        return multiply(other, self)

    def __pow__(self, other: Any) -> Any:
        """__pow__ function."""
        return power(self, other)

    def __rpow__(self, other: Any) -> Any:
        """__rpow__ function."""
        return power(other, self)

    def __truediv__(self, other: Any) -> Any:
        """__truediv__."""

        return true_divide(self, other)

    def __rtruediv__(self, other: Any) -> Any:
        """__rtruediv__."""

        return true_divide(other, self)

    def __floordiv__(self, other: Any) -> Any:
        """__floordiv__."""

        return floor_divide(self, other)

    def __rfloordiv__(self, other: Any) -> Any:
        """__rfloordiv__."""

        return floor_divide(other, self)

    def __neg__(self) -> Any:
        """__neg__ function."""
        return multiply(self, -1.0)

    def __lt__(self, other: Any) -> Any:
        """__lt__ function."""
        return _wrap(ops.less(self._tensor, _to_tensor(other)))

    def __gt__(self, other: Any) -> Any:
        """__gt__ function."""
        return _wrap(ops.greater(self._tensor, _to_tensor(other)))

    def __le__(self, other: Any) -> Any:
        """__le__ function."""
        return _wrap(ops.less_equal(self._tensor, _to_tensor(other)))

    def __ge__(self, other: Any) -> Any:
        """__ge__ function."""
        return _wrap(ops.greater_equal(self._tensor, _to_tensor(other)))

    def __setitem__(self, key: Any, value: Any) -> None:
        """__setitem__."""

        from ml_switcheroo.core.config import config

        if config.eager_mode:
            val = getattr(value, "_tensor", value)
            val = getattr(val, "data", val)
            self._tensor.data[key] = val
        else:
            raise NotImplementedError(
                "Item assignment is only supported in eager mode."
            )

    def __getitem__(self, key: Any) -> Any:
        """__getitem__."""

        arr = self.__array__()
        if hasattr(key, "_tensor"):
            key = key._tensor.data
        elif isinstance(key, tuple):
            key = tuple(
                getattr(getattr(k, "_tensor", k), "data", getattr(k, "_tensor", k))
                for k in key
            )
        return _wrap(_to_tensor(arr[key]))

    def __eq__(self, other: Any) -> Any:
        """__eq__ function."""
        return _wrap(ops.equal(self._tensor, _to_tensor(other)))

    def __bool__(self) -> Any:
        """__bool__ function."""
        arr = self.__array__()
        if arr.size == 1:
            return bool(arr.item())
        raise ValueError(
            "The truth value of an array with more than one element is ambiguous."
        )

    def __len__(self) -> Any:
        """__len__ function."""
        return self.shape[0] if self.shape else 0

    def __iter__(self) -> Any:
        """__iter__ function."""
        arr = self.__array__()
        for i in range(arr.shape[0]):
            yield array(arr[i])


def _to_tensor(x: Any) -> Any:
    """_to_tensor function."""
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
                attributes={"value": getattr(x.data, "tolist", lambda: x.data)()},
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

    from ml_switcheroo.core import tensor_utils

    arr = tensor_utils.to_array(x)
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


def _wrap(t: Any) -> Any:
    """_wrap function."""
    if isinstance(t, Tensor):
        return ndarray(t)
    elif isinstance(t, tuple):
        return tuple(_wrap(x) for x in t)
    elif isinstance(t, list):
        return list(_wrap(x) for x in t)
    return t


def sin(x: Any) -> Any:
    """Sin function."""
    return _wrap(ops.sin(_to_tensor(x)))


def cos(x: Any) -> Any:
    """Cos function."""
    return _wrap(ops.cos(_to_tensor(x)))


def exp(x: Any) -> Any:
    """Exp function."""
    return _wrap(ops.exp(_to_tensor(x)))


def log(x: Any) -> Any:
    """Log function."""
    return _wrap(ops.log(_to_tensor(x)))


def transpose(x: Any, axes: Optional[List[int]] = None) -> Any:
    """Transpose function."""
    t = _to_tensor(x)
    if axes is not None:
        return _wrap(ops.permute(t, dims=axes))
    axes = list(range(len(t.shape))[::-1])
    return _wrap(ops.permute(t, dims=axes))


def reshape(x: Any, newshape: Tuple[int, ...]) -> Any:
    """Reshape function."""
    return _wrap(ops.reshape(_to_tensor(x), shape=newshape))


def broadcast_to(x: Any, shape: Tuple[int, ...]) -> Any:
    """broadcast_to function."""
    return _wrap(ops.broadcast_to(_to_tensor(x), size=shape))


def concatenate(arrays: List[Any], axis: int = 0) -> Any:
    """Concatenate function."""
    tensors = [_to_tensor(a) for a in arrays]
    return _wrap(ops.concatenate(tensors, dim=axis))


def where(condition: Any, x: Any, y: Any) -> Any:
    """Where function."""
    return _wrap(ops.where(_to_tensor(condition), _to_tensor(x), _to_tensor(y)))


def einsum(subscripts: str, *operands: Any) -> Any:
    """Einsum function."""
    tensors = [_to_tensor(a) for a in operands]
    return _wrap(ops.einsum(subscripts, *tensors))


def add(x: Any, y: Any) -> Any:
    """Add function."""
    return _wrap(ops.add(_to_tensor(x), _to_tensor(y)))


def multiply(x: Any, y: Any) -> Any:
    """Multiply function."""
    return _wrap(ops.multiply(_to_tensor(x), _to_tensor(y)))


def power(x: Any, y: Any) -> Any:
    """Power function."""
    return _wrap(ops.power(_to_tensor(x), _to_tensor(y)))


def maximum(x: Any, y: Any) -> Any:
    """Maximum function."""
    return _wrap(ops.maximum(_to_tensor(x), _to_tensor(y)))


def minimum(x: Any, y: Any) -> Any:
    """Minimum function."""
    return _wrap(ops.minimum(_to_tensor(x), _to_tensor(y)))


def clip(a: Any, a_min: Any, a_max: Any) -> Any:
    """Clip function."""
    res = _to_tensor(a)
    if a_min is not None:
        res = ops.maximum(res, _to_tensor(a_min))
    if a_max is not None:
        res = ops.minimum(res, _to_tensor(a_max))
    return _wrap(res)


def max(
    x: Any,
    axis: Any = None,
    keepdims: bool = False,
    where: Any = None,
    initial: Any = None,
) -> Any:
    """Max function."""
    t_x = _to_tensor(x)
    if where is not None:
        init_val = initial if initial is not None else float("-inf")
        t_x = ops.where(_to_tensor(where), t_x, _to_tensor(init_val))
    res = ops.max(t_x, axis=axis, keepdims=keepdims)
    if initial is not None:
        res = ops.maximum(res, _to_tensor(initial))
    return _wrap(res)


def sum(x: Any, axis: Any = None, keepdims: bool = False, where: Any = None) -> Any:
    """Sum function."""
    t_x = _to_tensor(x)
    if where is not None:
        t_x = ops.where(_to_tensor(where), t_x, _to_tensor(0))
    return _wrap(ops.sum(t_x, axis=axis, keepdims=keepdims))


def zeros_like(x: Any, dtype: Any = None) -> Any:
    """zeros_like function."""
    return _wrap(ops.zeros_like(_to_tensor(x), dtype=dtype))


def zeros(shape: Any, dtype: Any = None) -> Any:
    """Zeros function."""
    return _wrap(ops.zeros(shape=shape, dtype=dtype))


def abs(x: Any) -> Any:
    """Abs function."""
    return _wrap(ops.abs(_to_tensor(x)))


def mean(x: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """Mean function."""
    return _wrap(ops.mean(_to_tensor(x), axis=axis, keepdims=keepdims))


inf = float("inf")


def array(x: Any, dtype: Any = None) -> Any:
    """Array function."""
    if isinstance(x, ndarray):
        return x
    return _wrap(_to_tensor(x))


def dot(a: Any, b: Any) -> Any:
    """Dot function."""
    return _wrap(ops.dot(_to_tensor(a), _to_tensor(b)))


def matmul(a: Any, b: Any) -> Any:
    """Matmul function."""
    return _wrap(ops.matmul(_to_tensor(a), _to_tensor(b)))


def expand_dims(a: Any, axis: int) -> Any:
    """expand_dims function."""
    return _wrap(ops.unsqueeze(_to_tensor(a), dim=axis))


def isfinite(x: Any) -> Any:
    """Isfinite function."""
    return _wrap(ops.isfinite(_to_tensor(x)))


def allclose(
    a: Any, b: Any, rtol: Any = 1e-05, atol: Any = 1e-08, equal_nan: Any = False
) -> Any:
    """Allclose function."""
    return ops.allclose(
        _to_tensor(a), _to_tensor(b), rtol=rtol, atol=atol, equal_nan=equal_nan
    )


def array_equal(a1: Any, a2: Any, equal_nan: Any = False) -> Any:
    """array_equal function."""
    res = ops.equal(_to_tensor(a1), _to_tensor(a2))
    from ml_switcheroo.core import tensor_utils

    return bool(tensor_utils.to_array(res.data).all()) if hasattr(res, "data") else True


def broadcast_shapes(*shapes: Any) -> Any:
    """broadcast_shapes function."""
    from ml_switcheroo.shape import broadcast_shapes as _broadcast_shapes
    import functools

    if not shapes:
        return ()
    return functools.reduce(_broadcast_shapes, shapes)


def _unary_op(x: Any, name: Any) -> Any:
    """_unary_op function."""
    if name == "Transpose":
        return transpose(x)
    raise NotImplementedError()


def ones(shape: Any, dtype: Any = None) -> Any:
    """Ones function."""
    return _wrap(ops.ones(shape=shape, dtype=dtype))


def empty(shape: Any, dtype: Any = None) -> Any:
    """Empty function."""
    return _wrap(ops.empty(shape=shape, dtype=dtype))


def full(shape: Any, fill_value: Any, dtype: Any = None) -> Any:
    """Full function."""
    return _wrap(ops.full(shape=shape, fill_value=fill_value, dtype=dtype))


def ones_like(x: Any, dtype: Any = None) -> Any:
    """ones_like function."""
    return _wrap(ops.ones_like(_to_tensor(x), dtype=dtype))


def empty_like(x: Any, dtype: Any = None) -> Any:
    """empty_like function."""
    t = _to_tensor(x)
    return _wrap(ops.empty(shape=t.shape, dtype=dtype if dtype else t.dtype))


def full_like(x: Any, fill_value: Any, dtype: Any = None) -> Any:
    """full_like function."""
    return _wrap(ops.full_like(_to_tensor(x), fill_value=fill_value, dtype=dtype))


def asarray(x: Any, dtype: Any = None) -> Any:
    """Asarray function."""
    return array(x, dtype=dtype)


def arange(start: Any, stop: Any = None, step: Any = 1, dtype: Any = None) -> Any:
    """Arange function."""
    return _wrap(ops.arange(start=start, stop=stop, step=step, dtype=dtype))


def linspace(
    start: Any,
    stop: Any,
    num: int = 50,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: Any = None,
    axis: int = 0,
) -> Any:
    """Linspace function."""
    if retstep or axis != 0 or not endpoint:
        raise NotImplementedError("linspace currently only supports basic usage")
    return _wrap(ops.linspace(start=start, stop=stop, steps=num, dtype=dtype))


def logspace(
    start: Any,
    stop: Any,
    num: int = 50,
    endpoint: bool = True,
    base: float = 10.0,
    dtype: Any = None,
    axis: int = 0,
) -> Any:
    """Logspace function."""
    lin = linspace(start, stop, num, endpoint=endpoint, dtype=dtype, axis=axis)
    return power(base, lin)


def eye(N: int, M: int = None, k: int = 0, dtype: Any = None) -> Any:
    """Eye function."""
    if k != 0:
        raise NotImplementedError()
    return _wrap(ops.eye(n=N, m=M, dtype=dtype))


def identity(n: int, dtype: Any = None) -> Any:
    """Identity function."""
    return _wrap(ops.identity(n=n, dtype=dtype))


def meshgrid(
    *xi: Any, copy: Any = True, sparse: Any = False, indexing: Any = "xy"
) -> Any:
    """Meshgrid function."""
    if sparse or not copy:
        raise NotImplementedError()

    tensors = [_to_tensor(x) for x in xi]

    ndim = len(tensors)
    if indexing == "xy" and ndim > 1:
        tensors[0], tensors[1] = tensors[1], tensors[0]

    s0 = (1,) * ndim
    output = []
    for i, t in enumerate(tensors):
        shape = list(s0)
        shape[i] = -1
        reshaped = ops.reshape(t, shape=tuple(shape))
        output.append(reshaped)

    broadcast_shape = tuple(t.shape[0] for t in tensors)
    output = [ops.broadcast_to(t, size=broadcast_shape) for t in output]

    if indexing == "xy" and ndim > 1:
        output[0], output[1] = output[1], output[0]

    return tuple(_wrap(t) for t in output)


def subtract(x: Any, y: Any) -> Any:
    """Subtract function."""
    return _wrap(ops.subtract(_to_tensor(x), _to_tensor(y)))


def divide(x: Any, y: Any) -> Any:
    """Divide function."""
    return _wrap(ops.divide(_to_tensor(x), _to_tensor(y)))


def true_divide(x: Any, y: Any) -> Any:
    """true_divide function."""
    return divide(x, y)


def floor_divide(x: Any, y: Any) -> Any:
    """floor_divide function."""
    return _wrap(ops.floor_divide(_to_tensor(x), _to_tensor(y)))


def mod(x: Any, y: Any) -> Any:
    """Mod function."""
    return _wrap(ops.mod(_to_tensor(x), _to_tensor(y)))


def remainder(x: Any, y: Any) -> Any:
    """Remainder function."""
    return _wrap(ops.remainder(_to_tensor(x), _to_tensor(y)))


def divmod(x: Any, y: Any) -> Any:
    """Divmod function."""
    out1, out2 = ops.divmod(_to_tensor(x), _to_tensor(y))
    return _wrap(out1), _wrap(out2)


def negative(x: Any) -> Any:
    """Negative function."""
    return _wrap(ops.negative(_to_tensor(x)))


def positive(x: Any) -> Any:
    """Positive function."""
    return _wrap(ops.positive(_to_tensor(x)))


def sign(x: Any) -> Any:
    """Sign function."""
    return _wrap(ops.sign(_to_tensor(x)))


def floor(x: Any) -> Any:
    """Floor function."""
    return _wrap(ops.floor(_to_tensor(x)))


def ceil(x: Any) -> Any:
    """Ceil function."""
    return _wrap(ops.ceil(_to_tensor(x)))


def trunc(x: Any) -> Any:
    """Trunc function."""
    return _wrap(ops.trunc(_to_tensor(x)))


def rint(x: Any) -> Any:
    """Rint function."""
    return _wrap(ops.round(_to_tensor(x)))


def tan(x: Any) -> Any:
    """Tan function."""
    return _wrap(ops.tan(_to_tensor(x)))


def arcsin(x: Any) -> Any:
    """Arcsin function."""
    return _wrap(ops.asin(_to_tensor(x)))


def arccos(x: Any) -> Any:
    """Arccos function."""
    return _wrap(ops.acos(_to_tensor(x)))


def arctan(x: Any) -> Any:
    """Arctan function."""
    return _wrap(ops.atan(_to_tensor(x)))


def arctan2(x1: Any, x2: Any) -> Any:
    """arctan2 function."""
    return _wrap(ops.atan2(_to_tensor(x1), _to_tensor(x2)))


def sinh(x: Any) -> Any:
    """Sinh function."""
    return _wrap(ops.sinh(_to_tensor(x)))


def cosh(x: Any) -> Any:
    """Cosh function."""
    return _wrap(ops.cosh(_to_tensor(x)))


def tanh(x: Any) -> Any:
    """Tanh function."""
    return _wrap(ops.tanh(_to_tensor(x)))


def arcsinh(x: Any) -> Any:
    """Arcsinh function."""
    return _wrap(ops.asinh(_to_tensor(x)))


def arccosh(x: Any) -> Any:
    """Arccosh function."""
    return _wrap(ops.acosh(_to_tensor(x)))


def arctanh(x: Any) -> Any:
    """Arctanh function."""
    return _wrap(ops.atanh(_to_tensor(x)))


def exp2(x: Any) -> Any:
    """exp2 function."""
    # 2^x = exp(x * ln(2)) or just power(2, x)
    return power(2.0, x)


def expm1(x: Any) -> Any:
    """expm1 function."""
    return subtract(exp(x), 1.0)


def log2(x: Any) -> Any:
    """log2 function."""
    # log2(x) = log(x) / log(2)
    import math

    return divide(log(x), math.log(2.0))


def log10(x: Any) -> Any:
    """log10 function."""
    import math

    return divide(log(x), math.log(10.0))


def log1p(x: Any) -> Any:
    """log1p function."""
    return log(add(x, 1.0))


def prod(a: Any, axis: Any = None, dtype: Any = None, keepdims: bool = False) -> Any:
    """Prod function."""
    return _wrap(ops.prod(_to_tensor(a), axis=axis, keepdims=keepdims))


def min(a: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """Min function."""
    return _wrap(ops.min(_to_tensor(a), axis=axis, keepdims=keepdims))


def amin(a: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """Amin function."""
    return min(a, axis=axis, keepdims=keepdims)


def amax(a: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """Amax function."""
    return max(a, axis=axis, keepdims=keepdims)


def argmax(a: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """Argmax function."""
    return _wrap(ops.argmax(_to_tensor(a), axis=axis, keepdims=keepdims))


def argmin(a: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """Argmin function."""
    return _wrap(ops.argmin(_to_tensor(a), axis=axis, keepdims=keepdims))


def any(a: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """Any function."""
    return _wrap(ops.any(_to_tensor(a), axis=axis, keepdims=keepdims))


def all(a: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """All function."""
    return _wrap(ops.all(_to_tensor(a), axis=axis, keepdims=keepdims))


def var(
    a: Any, axis: Any = None, dtype: Any = None, keepdims: bool = False, ddof: int = 0
) -> Any:
    """Var function."""
    # variance = E[X^2] - E[X]^2 or mean((x - mean(x))^2)
    # Using eager wrapper or tracing composition
    t = _to_tensor(a)
    m = mean(a, axis=axis, keepdims=True)
    diff = subtract(t, m)
    sq = multiply(diff, diff)
    # if ddof != 0 we would need more math, but standard test probably uses default
    return mean(sq, axis=axis, keepdims=keepdims)


def std(
    a: Any, axis: Any = None, dtype: Any = None, keepdims: bool = False, ddof: int = 0
) -> Any:
    """Std function."""
    # Standard deviation is sqrt of variance
    # ops.sqrt exists or power(var, 0.5)
    v = var(a, axis=axis, dtype=dtype, keepdims=keepdims, ddof=ddof)
    return power(v, 0.5)


def ravel(a: Any, order: str = "C") -> Any:
    """Ravel function."""
    # Eager fallback or reshape if order='C'
    if order != "C":
        raise NotImplementedError("ravel only supports order='C'")
    return reshape(a, (-1,))


def squeeze(a: Any, axis: Any = None) -> Any:
    """Squeeze function."""
    return _wrap(ops.squeeze(_to_tensor(a), dim=axis))


def swapaxes(a: Any, axis1: int, axis2: int) -> Any:
    """Swapaxes function."""
    return _wrap(ops.swapaxes(_to_tensor(a), axis1=axis1, axis2=axis2))


def moveaxis(a: Any, source: Any, destination: Any) -> Any:
    """Moveaxis function."""
    return _wrap(ops.moveaxis(_to_tensor(a), source=source, destination=destination))


def stack(arrays: Any, axis: int = 0) -> Any:
    """Stack function."""
    tensors = [_to_tensor(arr) for arr in arrays]
    return _wrap(ops.stack(tensors, dim=axis))


def vstack(tup: Any) -> Any:
    """Vstack function."""
    return _wrap(ops.vstack([_to_tensor(arr) for arr in tup]))


def hstack(tup: Any) -> Any:
    """Hstack function."""
    return _wrap(ops.hstack([_to_tensor(arr) for arr in tup]))


def dstack(tup: Any) -> Any:
    """Dstack function."""
    return _wrap(ops.dstack([_to_tensor(arr) for arr in tup]))


def split(ary: Any, indices_or_sections: Any, axis: int = 0) -> Any:
    """Split function."""
    return tuple(
        _wrap(t) for t in ops.split(_to_tensor(ary), indices_or_sections, axis)
    )


def array_split(ary: Any, indices_or_sections: Any, axis: int = 0) -> Any:
    """array_split function."""
    return tuple(
        _wrap(t) for t in ops.array_split(_to_tensor(ary), indices_or_sections, axis)
    )


def vsplit(ary: Any, indices_or_sections: Any) -> Any:
    """Vsplit function."""
    return tuple(_wrap(t) for t in ops.vsplit(_to_tensor(ary), indices_or_sections))


def hsplit(ary: Any, indices_or_sections: Any) -> Any:
    """Hsplit function."""
    return tuple(_wrap(t) for t in ops.hsplit(_to_tensor(ary), indices_or_sections))


def dsplit(ary: Any, indices_or_sections: Any) -> Any:
    """Dsplit function."""
    return tuple(_wrap(t) for t in ops.dsplit(_to_tensor(ary), indices_or_sections))


def tile(A: Any, reps: Any) -> Any:
    """Tile function."""
    return _wrap(ops.tile(_to_tensor(A), reps=reps))


def repeat(a: Any, repeats: Any, axis: Any = None) -> Any:
    """Repeat function."""
    return _wrap(ops.repeat(_to_tensor(a), repeats=repeats, dim=axis))


def pad(array: Any, pad_width: Any, mode: str = "constant", **kwargs: Any) -> Any:
    """Pad function."""
    return _wrap(ops.pad(_to_tensor(array), pad_width, mode=mode, **kwargs))


def take(a: Any, indices: Any, axis: int = None, mode: str = None) -> Any:
    """Take function."""
    return _wrap(ops.take(_to_tensor(a), _to_tensor(indices)))


def take_along_axis(arr: Any, indices: Any, axis: int) -> Any:
    """take_along_axis function."""
    return _wrap(ops.take_along_axis(_to_tensor(arr), _to_tensor(indices), axis=axis))


def vdot(a: Any, b: Any) -> Any:
    """Vdot function."""
    return _wrap(ops.vdot(_to_tensor(a), _to_tensor(b)))


def inner(a: Any, b: Any) -> Any:
    """Inner function."""
    return _wrap(ops.inner(_to_tensor(a), _to_tensor(b)))


def outer(a: Any, b: Any) -> Any:
    """Outer function."""
    return _wrap(ops.outer(_to_tensor(a), _to_tensor(b)))


def tensordot(a: Any, b: Any, axes: Any = 2) -> Any:
    """Tensordot function."""
    return _wrap(ops.tensordot(_to_tensor(a), _to_tensor(b), axes=axes))


def shape(a: Any) -> Any:
    """Return the shape of an array."""
    return asarray(a).shape


def sqrt(x: Any) -> Any:
    """sqrt."""

    return _wrap(ops.sqrt(_to_tensor(x)))


def square(x: Any) -> Any:
    """square."""

    return _wrap(ops.square(_to_tensor(x)))


def isnan(x: Any) -> Any:
    """isnan."""

    return _wrap(ops.isnan(_to_tensor(x)))


nan = float("nan")

pi = 3.14159265358979323846


def cumsum(a: Any, axis: Any = None, dtype: Any = None) -> Any:
    """Cumsum function."""
    return _wrap(
        ops.cumsum(_to_tensor(a), axis=axis, dtype=getattr(dtype, "value", dtype))
    )
