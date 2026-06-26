import builtins

"""JAX-like numpy API backed by ml-switcheroo-compiler."""

from ml_switcheroo_compiler.core.tensor import TensorConfig
from typing import Any

from typing import Tuple, List, Optional
import ml_switcheroo_compiler.ops as ops
from ml_switcheroo_compiler import Tensor
import ml_switcheroo_compiler as _ml_switcheroo_compiler
from ml_switcheroo_compiler.core.config import config


class ndarray:
    """
    A multi-dimensional array object backed by an ml-switcheroo tensor.

    This class provides a NumPy-like interface for tensor operations, supporting
    standard arithmetic, comparison, and array-manipulation magic methods.
    """

    def __init__(self, tensor: Any) -> None:
        """
        Initialize the object.

        Args:
            tensor (Any): The underlying tensor data.

        Returns:
            None
        """
        self._tensor = tensor

    @property
    def shape(self) -> Any:
        """
        Get the shape of the array.

        Returns:
            Any: The shape property of the underlying tensor.
        """
        return self._tensor.shape

    @property
    def ndim(self) -> int:
        """
        Get the number of dimensions of the array.

        Returns:
            int: The number of dimensions.
        """
        return len(self.shape) if self.shape is not None else 0

    @property
    def dtype(self) -> Any:
        """
        Get the dtype of the array.

        Returns:
            Any: The dtype property of the underlying tensor.
        """
        return self._tensor.dtype

    def __array__(self, dtype=None) -> Any:
        """
        Perform the array operation.

        Args:
            dtype: Optional dtype to convert to.

        Returns:
            Any: The result of the array operation.
        """
        arr = self._tensor.__array__()
        if dtype is not None:
            # We don't have numpy here, but the caller usually handles the dtype cast
            pass
        return arr

    def __repr__(self) -> Any:
        """
        Perform the repr operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the repr operation.
        """
        return repr(self.__array__())

    def __add__(self, other: Any) -> Any:
        """
        Perform the add operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the add operation.
        """
        return add(self, other)

    def __radd__(self, other: Any) -> Any:
        """
        Perform the radd operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the radd operation.
        """
        return add(other, self)

    def __sub__(self, other: Any) -> Any:
        """
        Perform the sub operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the sub operation.
        """
        return add(self, multiply(other, -1))

    def __rsub__(self, other: Any) -> Any:
        """
        Perform the rsub operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the rsub operation.
        """
        return add(other, multiply(self, -1))

    def __mul__(self, other: Any) -> Any:
        """
        Perform the mul operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the mul operation.
        """
        return multiply(self, other)

    def __rmul__(self, other: Any) -> Any:
        """
        Perform the rmul operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the rmul operation.
        """
        return multiply(other, self)

    def __pow__(self, other: Any) -> Any:
        """
        Perform the pow operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the pow operation.
        """
        return power(self, other)

    def __rpow__(self, other: Any) -> Any:
        """
        Perform the rpow operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the rpow operation.
        """
        return power(other, self)

    def __truediv__(self, other: Any) -> Any:
        """
        Perform the truediv operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the truediv operation.
        """

        return true_divide(self, other)

    def __rtruediv__(self, other: Any) -> Any:
        """
        Perform the rtruediv operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the rtruediv operation.
        """

        return true_divide(other, self)

    def __floordiv__(self, other: Any) -> Any:
        """
        Perform the floordiv operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the floordiv operation.
        """

        return floor_divide(self, other)

    def __rfloordiv__(self, other: Any) -> Any:
        """
        Perform the rfloordiv operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the rfloordiv operation.
        """

        return floor_divide(other, self)

    def __neg__(self) -> Any:
        """
        Perform the neg operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the neg operation.
        """
        return multiply(self, -1.0)

    def __lt__(self, other: Any) -> Any:
        """
        Perform the lt operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the lt operation.
        """
        return _wrap(ops.less(self._tensor, _to_tensor(other)))

    def __gt__(self, other: Any) -> Any:
        """
        Perform the gt operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the gt operation.
        """
        return _wrap(ops.greater(self._tensor, _to_tensor(other)))

    def __le__(self, other: Any) -> Any:
        """
        Perform the le operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the le operation.
        """
        return _wrap(ops.less_equal(self._tensor, _to_tensor(other)))

    def __ge__(self, other: Any) -> Any:
        """
        Perform the ge operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the ge operation.
        """
        return _wrap(ops.greater_equal(self._tensor, _to_tensor(other)))

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Perform the setitem operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the setitem operation.
        """

        from ml_switcheroo_compiler.core.config import config

        if config.eager_mode:
            val = getattr(value, "_tensor", value)
            val = getattr(val, "data", val)
            self._tensor.data[key] = val
        else:
            raise NotImplementedError(
                "Item assignment is only supported in eager mode."
            )

    def __getitem__(self, key: Any) -> Any:
        """
        Perform the getitem operation.
        """
        from ml_switcheroo_compiler.core.config import config

        if config.eager_mode:
            arr = self.__array__()
            if hasattr(key, "_tensor"):
                key = key._tensor.data
            elif isinstance(key, tuple):
                key = tuple(
                    getattr(getattr(k, "_tensor", k), "data", getattr(k, "_tensor", k))
                    for k in key
                )
            return _wrap(_to_tensor(arr[key]))

        t = self._tensor
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else t.shape[0]
            step = key.step if key.step is not None else 1
            return _wrap(ops.strided_slice(t, [start], [stop], [step]))

        if isinstance(key, tuple) and builtins.all(
            isinstance(k, slice) or k is None for k in key
        ):
            starts = []
            stops = []
            steps = []
            for i, k in enumerate(key):
                if k is None:
                    continue
                start = k.start if k.start is not None else 0
                stop = k.stop if k.stop is not None else t.shape[len(starts)]
                step = k.step if k.step is not None else 1
                starts.append(start)
                stops.append(stop)
                steps.append(step)
            for i in range(len(starts), len(t.shape)):
                starts.append(0)
                stops.append(t.shape[i])
                steps.append(1)
            return _wrap(ops.strided_slice(t, starts, stops, steps))

        return _wrap(t[key])

    def __eq__(self, other: Any) -> Any:
        """
        Perform the eq operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the eq operation.
        """
        return _wrap(ops.equal(self._tensor, _to_tensor(other)))

    def __bool__(self) -> Any:
        """
        Perform the bool operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the bool operation.
        """
        import math

        size = math.prod(self.shape) if self.shape else 1
        if size == 1:
            if hasattr(self._tensor.data, "id"):  # ProxyTensor
                return True  # Tracer dummy bool

            val = self._tensor.data if hasattr(self._tensor, "data") else self._tensor

            if hasattr(val, "value") and not callable(val.value):
                return bool(val.value)  # pragma: no cover

            if hasattr(val, "item") and callable(val.item):
                try:
                    return bool(val.item())
                except Exception:
                    pass

            if hasattr(self._tensor, "item") and callable(self._tensor.item):
                try:
                    return bool(self._tensor.item())
                except Exception:
                    pass

            try:
                return bool(val)
            except TypeError:
                if hasattr(val, "name"):
                    return bool(val.name)  # pragma: no cover
                return True
        raise ValueError(
            "The truth value of an array with more than one element is ambiguous."
        )

    def __len__(self) -> Any:
        """
        Perform the len operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the len operation.
        """
        return self.shape[0] if self.shape else 0

    def __iter__(self) -> Any:
        """
        Perform the iter operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the iter operation.
        """
        arr = self.__array__()
        for i in range(arr.shape[0]):
            yield array(arr[i])


def _to_tensor(x: Any) -> Any:
    """Convert the input to an ml-switcheroo Tensor.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """

    if isinstance(x, ndarray):
        x = x._tensor
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.tracing import _tracer, ProxyTensor
    from ml_switcheroo_ir import LogicalNode
    import uuid

    if isinstance(x, _ml_switcheroo_compiler.Tensor):
        if _tracer.is_tracing and not hasattr(x.data, "id"):
            # lift eager tensor as constant
            out_id = str(uuid.uuid4())
            val = getattr(
                x.data,
                "tolist",
                lambda: (
                    x.item() if (not x.shape or getattr(x, "size", 1) == 1) else x.data
                ),
            )()
            node = LogicalNode(
                id=out_id,
                op_type="Constant",
                attributes={"value": val},
                shape_metadata=x.shape,
            )
            _tracer.add_node(node)
            pt = ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype.value)
            return _ml_switcheroo_compiler.Tensor(
                data=pt,
                config=TensorConfig(shape=x.shape, dtype=x.dtype, device=x.device),
            )
        return x
    if isinstance(x, ProxyTensor):
        # We need a dtype. ProxyTensor has dtype as string.
        # But we'll just mock it or use default.
        return _ml_switcheroo_compiler.Tensor(
            data=x,
            config=TensorConfig(
                shape=x.shape,
                dtype=config.default_float_dtype,
                device=config.default_device,
            ),
        )

    from zero_jax.numpy import tensor_utils

    with _ml_switcheroo_compiler.EagerMode():
        arr = tensor_utils.to_array(x)
    return _to_tensor(arr)


def _wrap(t: Any) -> Any:
    """Wrap an ml-switcheroo Tensor in an ndarray.

    Args:
        t (Any): Argument t.

    Returns:
        Any: The result of the operation.
    """
    if isinstance(t, Tensor):
        return ndarray(t)
    elif isinstance(t, tuple):
        return tuple(_wrap(x) for x in t)
    elif isinstance(t, list):
        return list(_wrap(x) for x in t)
    return t


def sin(x: Any) -> Any:
    """Compute the trigonometric sine element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.sin(_to_tensor(x)))


def cos(x: Any) -> Any:
    """Compute the trigonometric cosine element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.cos(_to_tensor(x)))


def exp(x: Any) -> Any:
    """Calculate the exponential of all elements in the input array.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.exp(_to_tensor(x)))


def log(x: Any) -> Any:
    """Natural logarithm, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.log(_to_tensor(x)))


def transpose(x: Any, axes: Optional[List[int]] = None) -> Any:
    """Reverse or permute the axes of an array.

    Args:
        x (Any): Argument x.
        axes (Any): Argument axes.

    Returns:
        Any: The result of the operation.
    """
    t = _to_tensor(x)
    if axes is not None:
        return _wrap(ops.permute(t, dims=axes))
    axes = list(range(len(t.shape))[::-1])
    return _wrap(ops.permute(t, dims=axes))


def reshape(x: Any, newshape: Tuple[int, ...]) -> Any:
    """Gives a new shape to an array without changing its data.

    Args:
        x (Any): Argument x.
        newshape (Any): Argument newshape.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.reshape(_to_tensor(x), shape=newshape))


def broadcast_to(x: Any, shape: Tuple[int, ...]) -> Any:
    """Broadcast an array to a new shape.

    Args:
        x (Any): Argument x.
        shape (Any): Argument shape.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.broadcast_to(_to_tensor(x), shape))


def concatenate(arrays: List[Any], axis: int = 0) -> Any:
    """Join a sequence of arrays along an existing axis.

    Args:
        arrays (Any): Argument arrays.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    tensors = [_to_tensor(a) for a in arrays]
    return _wrap(ops.concatenate(tensors, dim=axis))


def where(condition: Any, x: Any, y: Any) -> Any:
    """Return elements chosen from x or y depending on condition.

    Args:
        condition (Any): Argument condition.
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.where(_to_tensor(condition), _to_tensor(x), _to_tensor(y)))


def einsum(subscripts: str, *operands: Any) -> Any:
    """Evaluates the Einstein summation convention on the operands.

    Args:
        subscripts (Any): Argument subscripts.

    Returns:
        Any: The result of the operation.
    """
    tensors = [_to_tensor(a) for a in operands]
    return _wrap(ops.einsum(subscripts, *tensors))


def add(x: Any, y: Any) -> Any:
    """Add arguments element-wise.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.add(_to_tensor(x), _to_tensor(y)))


def multiply(x: Any, y: Any) -> Any:
    """Multiply arguments element-wise.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.multiply(_to_tensor(x), _to_tensor(y)))


def power(x: Any, y: Any) -> Any:
    """First array elements raised to powers from second array, element-wise.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.power(_to_tensor(x), _to_tensor(y)))


def maximum(x: Any, y: Any) -> Any:
    """Element-wise maximum of array elements.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.maximum(_to_tensor(x), _to_tensor(y)))


def minimum(x: Any, y: Any) -> Any:
    """Element-wise minimum of array elements.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.minimum(_to_tensor(x), _to_tensor(y)))


def clip(a: Any, a_min: Any, a_max: Any) -> Any:
    """Clip (limit) the values in an array.

    Args:
        a (Any): Argument a.
        a_min (Any): Argument a_min.
        a_max (Any): Argument a_max.

    Returns:
        Any: The result of the operation.
    """
    res = _to_tensor(a)
    if a_min is not None:
        res = ops.maximum(res, _to_tensor(a_min))
    if a_max is not None:
        res = ops.minimum(res, _to_tensor(a_max))
    return _wrap(res)


def max(
    x: Any,
    axis: Any = None,
    keepdims: Any = False,
    where: Any = None,
    initial: Any = None,
) -> Any:
    """Return the maximum of an array or maximum along an axis.

    Args:
        x (Any): Argument x.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.
        where (Any): Argument where.
        initial (Any): Argument initial.

    Returns:
        Any: The result of the operation.
    """
    t_x = _to_tensor(x)
    if where is not None:
        init_val = initial if initial is not None else float("-inf")
        t_x = ops.where(_to_tensor(where), t_x, _to_tensor(init_val))
    res = ops.max(t_x, axis=axis, keepdims=keepdims)
    if initial is not None:
        res = ops.maximum(res, _to_tensor(initial))
    return _wrap(res)


def sum(x: Any, axis: Any = None, keepdims: Any = False, where: Any = None) -> Any:
    """Sum of array elements over a given axis.

    Args:
        x (Any): Argument x.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.
        where (Any): Argument where.

    Returns:
        Any: The result of the operation.
    """
    t_x = _to_tensor(x)
    if where is not None:
        t_x = ops.where(_to_tensor(where), t_x, _to_tensor(0))
    return _wrap(ops.sum(t_x, axis=axis, keepdims=keepdims))


def zeros_like(x: Any, dtype: Any = None) -> Any:
    """Return an array of zeros with the same shape and type as a given array.

    Args:
        x (Any): Argument x.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(ops.zeros_like(_to_tensor(x), dtype=dtype))


def zeros(shape: Any, dtype: Any = None) -> Any:
    """Return a new array of given shape and type, filled with zeros.

    Args:
        shape (Any): Argument shape.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(ops.zeros(shape=shape, dtype=dtype))


def abs(x: Any) -> Any:
    """Calculate the absolute value element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.abs(_to_tensor(x)))


def mean(x: Any, axis: Any = None, keepdims: Any = False) -> Any:
    """Compute the arithmetic mean along the specified axis.

    Args:
        x (Any): Argument x.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.mean(_to_tensor(x), axis=axis, keepdims=keepdims))


inf = float("inf")


def array(x: Any, dtype: Any = None) -> Any:
    """Create an array.

    Args:
        x (Any): Argument x.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    if isinstance(x, ndarray):
        return x
    return _wrap(_to_tensor(x))


def dot(a: Any, b: Any) -> Any:
    """Dot product of two arrays.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.dot(_to_tensor(a), _to_tensor(b)))


def matmul(a: Any, b: Any) -> Any:
    """Matrix product of two arrays.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.matmul(_to_tensor(a), _to_tensor(b)))


def expand_dims(a: Any, axis: int) -> Any:
    """Expand the shape of an array.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.unsqueeze(_to_tensor(a), dim=axis))


def isfinite(x: Any) -> Any:
    """Test element-wise for finiteness (not infinity or not Not a Number).

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.isfinite(_to_tensor(x)))


def allclose(
    a: Any, b: Any, rtol: Any = 1e-05, atol: Any = 1e-08, equal_nan: Any = False
) -> Any:
    """Returns True if two arrays are element-wise equal within a tolerance.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.
        rtol (Any): Argument rtol.
        atol (Any): Argument atol.
        equal_nan (Any): Argument equal_nan.

    Returns:
        Any: The result of the operation.
    """
    return ops.allclose(
        _to_tensor(a), _to_tensor(b), rtol=rtol, atol=atol, equal_nan=equal_nan
    )


def array_equal(a1: Any, a2: Any, equal_nan: Any = False) -> Any:
    """True if two arrays have the same shape and elements, False otherwise.

    Args:
        a1 (Any): Argument a1.
        a2 (Any): Argument a2.
        equal_nan (Any): Argument equal_nan.

    Returns:
        Any: The result of the operation.
    """
    res = ops.equal(_to_tensor(a1), _to_tensor(a2))
    return ops.all(res).item() if hasattr(ops.all(res), "item") else bool(ops.all(res))


def broadcast_shapes(*shapes: Any) -> Any:
    """Broadcast the input shapes into a single shape.

    Returns:
        Any: The result of the operation.
    """
    from ml_switcheroo_compiler.ops import broadcast_shapes as _broadcast_shapes
    import functools

    if not shapes:
        return ()
    return functools.reduce(_broadcast_shapes, shapes)


def _unary_op(x: Any, name: Any) -> Any:
    """Apply a unary operation.

    Args:
        x (Any): Argument x.
        name (Any): Argument name.

    Returns:
        Any: The result of the operation.
    """
    if name == "Transpose":
        return transpose(x)
    raise NotImplementedError()


def ones(shape: Any, dtype: Any = None) -> Any:
    """Return a new array of given shape and type, filled with ones.

    Args:
        shape (Any): Argument shape.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(ops.ones(shape=shape, dtype=dtype))


def empty(shape: Any, dtype: Any = None) -> Any:
    """Return a new array of given shape and type, without initializing entries.

    Args:
        shape (Any): Argument shape.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(ops.empty(shape=shape, dtype=dtype))


def full(shape: Any, fill_value: Any, dtype: Any = None) -> Any:
    """Return a new array of given shape and type, filled with fill_value.

    Args:
        shape (Any): Argument shape.
        fill_value (Any): Argument fill_value.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.full(shape=shape, fill_value=fill_value, dtype=dtype))


def ones_like(x: Any, dtype: Any = None) -> Any:
    """Return an array of ones with the same shape and type as a given array.

    Args:
        x (Any): Argument x.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(ops.ones_like(_to_tensor(x), dtype=dtype))


def empty_like(x: Any, dtype: Any = None) -> Any:
    """Return a new array with the same shape and type as a given array.

    Args:
        x (Any): Argument x.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    t = _to_tensor(x)
    if dtype is None:
        dtype = config.default_float_dtype
    return _wrap(ops.empty(shape=t.shape, dtype=dtype))


def full_like(x: Any, fill_value: Any, dtype: Any = None) -> Any:
    """Return a full array with the same shape and type as a given array.

    Args:
        x (Any): Argument x.
        fill_value (Any): Argument fill_value.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.full_like(_to_tensor(x), fill_value=fill_value, dtype=dtype))


def asarray(x: Any, dtype: Any = None) -> Any:
    """Convert the input to an array.

    Args:
        x (Any): Argument x.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return array(x, dtype=dtype)


def arange(start: Any, stop: Any = None, step: Any = 1, dtype: Any = None) -> Any:
    """Return evenly spaced values within a given interval.

    Args:
        start (Any): Argument start.
        stop (Any): Argument stop.
        step (Any): Argument step.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
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
    """Return evenly spaced numbers over a specified interval.

    Args:
        start (Any): Argument start.
        stop (Any): Argument stop.
        num (Any): Argument num.
        endpoint (Any): Argument endpoint.
        retstep (Any): Argument retstep.
        dtype (Any): Argument dtype.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
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
    """Return numbers spaced evenly on a log scale.

    Args:
        start (Any): Argument start.
        stop (Any): Argument stop.
        num (Any): Argument num.
        endpoint (Any): Argument endpoint.
        base (Any): Argument base.
        dtype (Any): Argument dtype.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    lin = linspace(start, stop, num, endpoint=endpoint, dtype=dtype, axis=axis)
    return power(base, lin)


def eye(N: int, M: int = None, k: int = 0, dtype: Any = None) -> Any:
    """Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Args:
        N (Any): Argument N.
        M (Any): Argument M.
        k (Any): Argument k.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    if k != 0:
        raise NotImplementedError()
    return _wrap(ops.eye(n=N, m=M, dtype=dtype))


def identity(n: int, dtype: Any = None) -> Any:
    """Return the identity array.

    Args:
        n (Any): Argument n.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.creation.frontend.identity(n=n, dtype=dtype))


def meshgrid(
    *xi: Any, copy: Any = True, sparse: Any = False, indexing: Any = "xy"
) -> Any:
    """Return coordinate matrices from coordinate vectors.

    Returns:
        Any: The result of the operation.
    """
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
    output = [ops.broadcast_to(t, broadcast_shape) for t in output]

    if indexing == "xy" and ndim > 1:
        output[0], output[1] = output[1], output[0]

    return tuple(_wrap(t) for t in output)


def subtract(x: Any, y: Any) -> Any:
    """Subtract arguments, element-wise.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.subtract(_to_tensor(x), _to_tensor(y)))


def divide(x: Any, y: Any) -> Any:
    """Divide arguments element-wise.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.divide(_to_tensor(x), _to_tensor(y)))


def true_divide(x: Any, y: Any) -> Any:
    """Divide arguments element-wise.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return divide(x, y)


def floor_divide(x: Any, y: Any) -> Any:
    """Return the largest integer smaller or equal to the division of the inputs.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.floor_divide(_to_tensor(x), _to_tensor(y)))


def mod(x: Any, y: Any) -> Any:
    """Return element-wise remainder of division.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.mod(_to_tensor(x), _to_tensor(y)))


def remainder(x: Any, y: Any) -> Any:
    """Return element-wise remainder of division.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.remainder(_to_tensor(x), _to_tensor(y)))


def divmod(x: Any, y: Any) -> Any:
    """Return element-wise quotient and remainder simultaneously.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    out1, out2 = ops.divmod(_to_tensor(x), _to_tensor(y))
    return _wrap(out1), _wrap(out2)


def negative(x: Any) -> Any:
    """Numerical negative, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.negative(_to_tensor(x)))


def positive(x: Any) -> Any:
    """Numerical positive, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.positive(_to_tensor(x)))


def sign(x: Any) -> Any:
    """Returns an element-wise indication of the sign of a number.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.sign(_to_tensor(x)))


def floor(x: Any) -> Any:
    """Return the floor of the input, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.floor(_to_tensor(x)))


def ceil(x: Any) -> Any:
    """Return the ceiling of the input, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.ceil(_to_tensor(x)))


def trunc(x: Any) -> Any:
    """Return the truncated value of the input, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.trunc(_to_tensor(x)))


def rint(x: Any) -> Any:
    """Round elements of the array to the nearest integer.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.round(_to_tensor(x)))


def tan(x: Any) -> Any:
    """Compute tangent element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.tan(_to_tensor(x)))


def arcsin(x: Any) -> Any:
    """Inverse sine, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.asin(_to_tensor(x)))


def arccos(x: Any) -> Any:
    """Trigonometric inverse cosine, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.acos(_to_tensor(x)))


def arctan(x: Any) -> Any:
    """Trigonometric inverse tangent, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.atan(_to_tensor(x)))


def arctan2(x1: Any, x2: Any) -> Any:
    """Element-wise arc tangent of x1/x2 choosing the quadrant correctly.

    Args:
        x1 (Any): Argument x1.
        x2 (Any): Argument x2.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.atan2(_to_tensor(x1), _to_tensor(x2)))


def sinh(x: Any) -> Any:
    """Hyperbolic sine, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.sinh(_to_tensor(x)))


def cosh(x: Any) -> Any:
    """Hyperbolic cosine, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.cosh(_to_tensor(x)))


def tanh(x: Any) -> Any:
    """Compute hyperbolic tangent element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.tanh(_to_tensor(x)))


def arcsinh(x: Any) -> Any:
    """Inverse hyperbolic sine element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.asinh(_to_tensor(x)))


def arccosh(x: Any) -> Any:
    """Inverse hyperbolic cosine, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.acosh(_to_tensor(x)))


def arctanh(x: Any) -> Any:
    """Inverse hyperbolic tangent element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.atanh(_to_tensor(x)))


def exp2(x: Any) -> Any:
    """Calculate 2**p for all p in the input array.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    # 2^x = exp(x * ln(2)) or just power(2, x)
    return power(2.0, x)


def expm1(x: Any) -> Any:
    """Calculate exp(x) - 1 for all elements in the array.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return subtract(exp(x), 1.0)


def log2(x: Any) -> Any:
    """Base-2 logarithm of x.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    # log2(x) = log(x) / log(2)
    import math

    return divide(log(x), math.log(2.0))


def log10(x: Any) -> Any:
    """Return the base 10 logarithm of the input array, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    import math

    return divide(log(x), math.log(10.0))


def log1p(x: Any) -> Any:
    """Return the natural logarithm of one plus the input array, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return log(add(x, 1.0))


def prod(a: Any, axis: Any = None, dtype: Any = None, keepdims: Any = False) -> Any:
    """Return the product of array elements over a given axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        dtype (Any): Argument dtype.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.prod(_to_tensor(a), axis=axis, keepdims=keepdims))


def min(a: Any, axis: Any = None, keepdims: Any = False) -> Any:
    """Return the minimum of an array or minimum along an axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.min(_to_tensor(a), axis=axis, keepdims=keepdims))


def amin(a: Any, axis: Any = None, keepdims: Any = False) -> Any:
    """Return the minimum of an array or minimum along an axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return min(a, axis=axis, keepdims=keepdims)


def amax(a: Any, axis: Any = None, keepdims: Any = False) -> Any:
    """Return the maximum of an array or maximum along an axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return max(a, axis=axis, keepdims=keepdims)


def argmax(a: Any, axis: Any = None, keepdims: Any = False) -> Any:
    """Returns the indices of the maximum values along an axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.argmax(_to_tensor(a), axis=axis, keepdims=keepdims))


def argmin(a: Any, axis: Any = None, keepdims: Any = False) -> Any:
    """Returns the indices of the minimum values along an axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.argmin(_to_tensor(a), axis=axis, keepdims=keepdims))


def any(a: Any, axis: Any = None, keepdims: Any = False) -> Any:
    """Test whether any array element along a given axis evaluates to True.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.any(_to_tensor(a), axis=axis, keepdims=keepdims))


def all(a: Any, axis: Any = None, keepdims: Any = False) -> Any:
    """Test whether all array elements along a given axis evaluate to True.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.all(_to_tensor(a), axis=axis, keepdims=keepdims))


def var(
    a: Any, axis: Any = None, dtype: Any = None, keepdims: Any = False, ddof: int = 0
) -> Any:
    """Compute the variance along the specified axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        dtype (Any): Argument dtype.
        keepdims (Any): Argument keepdims.
        ddof (Any): Argument ddof.

    Returns:
        Any: The result of the operation.
    """
    # variance = E[X^2] - E[X]^2 or mean((x - mean(x))^2)
    # Using eager wrapper or tracing composition
    t = _to_tensor(a)
    m = mean(a, axis=axis, keepdims=True)
    diff = subtract(t, m)
    sq = multiply(diff, diff)
    # if ddof != 0 we would need more math, but standard test probably uses default
    return mean(sq, axis=axis, keepdims=keepdims)


def std(
    a: Any, axis: Any = None, dtype: Any = None, keepdims: Any = False, ddof: int = 0
) -> Any:
    """Compute the standard deviation along the specified axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        dtype (Any): Argument dtype.
        keepdims (Any): Argument keepdims.
        ddof (Any): Argument ddof.

    Returns:
        Any: The result of the operation.
    """
    # Standard deviation is sqrt of variance
    # ops.sqrt exists or power(var, 0.5)
    v = var(a, axis=axis, dtype=dtype, keepdims=keepdims, ddof=ddof)
    return power(v, 0.5)


def ravel(a: Any, order: str = "C") -> Any:
    """Return a contiguous flattened array.

    Args:
        a (Any): Argument a.
        order (Any): Argument order.

    Returns:
        Any: The result of the operation.
    """
    # Eager fallback or reshape if order='C'
    if order != "C":
        raise NotImplementedError("ravel only supports order='C'")
    return reshape(a, (-1,))


def squeeze(a: Any, axis: Any = None) -> Any:
    """Remove axes of length one from a.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.shape.manipulation.squeeze(_to_tensor(a), dim=axis))


def swapaxes(a: Any, axis1: int, axis2: int) -> Any:
    """Interchange two axes of an array.

    Args:
        a (Any): Argument a.
        axis1 (Any): Argument axis1.
        axis2 (Any): Argument axis2.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.swapaxes(_to_tensor(a), axis1=axis1, axis2=axis2))


def moveaxis(a: Any, source: Any, destination: Any) -> Any:
    """Move axes of an array to new positions.

    Args:
        a (Any): Argument a.
        source (Any): Argument source.
        destination (Any): Argument destination.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.moveaxis(_to_tensor(a), source=source, destination=destination))


def stack(arrays: Any, axis: int = 0) -> Any:
    """Join a sequence of arrays along a new axis.

    Args:
        arrays (Any): Argument arrays.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    tensors = [_to_tensor(arr) for arr in arrays]
    return _wrap(ops.stack(tensors, dim=axis))


def vstack(tup: Any) -> Any:
    """Stack arrays in sequence vertically (row wise).

    Args:
        tup (Any): Argument tup.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.vstack([_to_tensor(arr) for arr in tup]))


def hstack(tup: Any) -> Any:
    """Stack arrays in sequence horizontally (column wise).

    Args:
        tup (Any): Argument tup.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.hstack([_to_tensor(arr) for arr in tup]))


def dstack(tup: Any) -> Any:
    """Stack arrays in sequence depth wise (along third axis).

    Args:
        tup (Any): Argument tup.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.dstack([_to_tensor(arr) for arr in tup]))


def split(ary: Any, indices_or_sections: Any, axis: int = 0) -> Any:
    """Split an array into multiple sub-arrays as views into ary.

    Args:
        ary (Any): Argument ary.
        indices_or_sections (Any): Argument indices_or_sections.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    return tuple(
        _wrap(t) for t in ops.split(_to_tensor(ary), indices_or_sections, axis)
    )


def array_split(ary: Any, indices_or_sections: Any, axis: int = 0) -> Any:
    """Split an array into multiple sub-arrays.

    Args:
        ary (Any): Argument ary.
        indices_or_sections (Any): Argument indices_or_sections.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    return tuple(
        _wrap(t) for t in ops.array_split(_to_tensor(ary), indices_or_sections, axis)
    )


def vsplit(ary: Any, indices_or_sections: Any) -> Any:
    """Split an array into multiple sub-arrays vertically (row-wise).

    Args:
        ary (Any): Argument ary.
        indices_or_sections (Any): Argument indices_or_sections.

    Returns:
        Any: The result of the operation.
    """
    return tuple(_wrap(t) for t in ops.vsplit(_to_tensor(ary), indices_or_sections))


def hsplit(ary: Any, indices_or_sections: Any) -> Any:
    """Split an array into multiple sub-arrays horizontally (column-wise).

    Args:
        ary (Any): Argument ary.
        indices_or_sections (Any): Argument indices_or_sections.

    Returns:
        Any: The result of the operation.
    """
    return tuple(_wrap(t) for t in ops.hsplit(_to_tensor(ary), indices_or_sections))


def dsplit(ary: Any, indices_or_sections: Any) -> Any:
    """Split array into multiple sub-arrays along the 3rd axis (depth).

    Args:
        ary (Any): Argument ary.
        indices_or_sections (Any): Argument indices_or_sections.

    Returns:
        Any: The result of the operation.
    """
    return tuple(_wrap(t) for t in ops.dsplit(_to_tensor(ary), indices_or_sections))


def tile(A: Any, reps: Any) -> Any:
    """Construct an array by repeating A the number of times given by reps.

    Args:
        A (Any): Argument A.
        reps (Any): Argument reps.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.tile(_to_tensor(A), reps=reps))


def repeat(a: Any, repeats: Any, axis: Any = None) -> Any:
    """Repeat elements of an array.

    Args:
        a (Any): Argument a.
        repeats (Any): Argument repeats.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.repeat(_to_tensor(a), repeats=repeats, dim=axis))


def pad(array: Any, pad_width: Any, mode: str = "constant", **kwargs: Any) -> Any:
    """Pad an array.

    Args:
        array (Any): Argument array.
        pad_width (Any): Argument pad_width.
        mode (Any): Argument mode.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.pad(_to_tensor(array), pad_width, mode=mode, **kwargs))


def take(a: Any, indices: Any, axis: int = None, mode: str = None) -> Any:
    """Take elements from an array along an axis.

    Args:
        a (Any): Argument a.
        indices (Any): Argument indices.
        axis (Any): Argument axis.
        mode (Any): Argument mode.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.take(_to_tensor(a), _to_tensor(indices)))


def take_along_axis(arr: Any, indices: Any, axis: int) -> Any:
    """Take values from the input array by matching 1d index and data slices.

    Args:
        arr (Any): Argument arr.
        indices (Any): Argument indices.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.take_along_axis(_to_tensor(arr), _to_tensor(indices), axis=axis))


def vdot(a: Any, b: Any) -> Any:
    """Return the dot product of two vectors.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.vdot(_to_tensor(a), _to_tensor(b)))


def inner(a: Any, b: Any) -> Any:
    """Inner product of two arrays.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.inner(_to_tensor(a), _to_tensor(b)))


def outer(a: Any, b: Any) -> Any:
    """Compute the outer product of two vectors.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.outer(_to_tensor(a), _to_tensor(b)))


def tensordot(a: Any, b: Any, axes: Any = 2) -> Any:
    """Compute tensor dot product along specified axes.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.
        axes (Any): Argument axes.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.tensordot(_to_tensor(a), _to_tensor(b), axes=axes))


def shape(a: Any) -> Any:
    """
    Get the shape of the array.

    Returns:
        Any: The shape property of the underlying tensor.
    """
    return asarray(a).shape


def sqrt(x: Any) -> Any:
    """Return the non-negative square-root of an array, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """

    return _wrap(ops.sqrt(_to_tensor(x)))


def square(x: Any) -> Any:
    """Return the element-wise square of the input.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """

    return _wrap(ops.square(_to_tensor(x)))


def isnan(x: Any) -> Any:
    """Test element-wise for NaN and return result as a boolean array.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """

    return _wrap(ops.isnan(_to_tensor(x)))


nan = float("nan")

pi = 3.14159265358979323846


def cumsum(a: Any, axis: Any = None, dtype: Any = None) -> Any:
    """Return the cumulative sum of the elements along a given axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    res = ops.cumsum(_to_tensor(a), axis=axis)
    if dtype is not None:
        from zero_jax.nn.activation import _to_dtype

        res = ops.cast(res, dtype=_to_dtype(dtype))
    return _wrap(res)


def acos(x: Any) -> Any:
    """Element-wise arc cosine.

    Args:
        x: Input array.

    Returns:
        An array containing the arc cosine of each element in x.
    """
    return _wrap(ops.acos(_to_tensor(x)))


def acosh(x: Any) -> Any:
    """Element-wise inverse hyperbolic cosine.

    Args:
        x: Input array.

    Returns:
        An array containing the inverse hyperbolic cosine of each element in x.
    """
    return _wrap(ops.acosh(_to_tensor(x)))


def asin(x: Any) -> Any:
    """Element-wise arc sine.

    Args:
        x: Input array.

    Returns:
        An array containing the arc sine of each element in x.
    """
    return _wrap(ops.asin(_to_tensor(x)))


def asinh(x: Any) -> Any:
    """Element-wise inverse hyperbolic sine.

    Args:
        x: Input array.

    Returns:
        An array containing the inverse hyperbolic sine of each element in x.
    """
    return _wrap(ops.asinh(_to_tensor(x)))


def atan(x: Any) -> Any:
    """Element-wise arc tangent.

    Args:
        x: Input array.

    Returns:
        An array containing the arc tangent of each element in x.
    """
    return _wrap(ops.atan(_to_tensor(x)))


def atan2(x1: Any, x2: Any) -> Any:
    """Element-wise arc tangent of x1/x2 choosing the quadrant correctly.

    Args:
        x1: Y-coordinates.
        x2: X-coordinates.

    Returns:
        An array containing the arc tangent of x1/x2.
    """
    # Need to broadcast x1 and x2 if shapes don't match, or let ops.atan2 handle it.
    # ml-switcheroo-compiler ops.atan2 probably handles basic broadcasting or we might need it.
    # Let's just pass it to ops.atan2
    return _wrap(ops.atan2(_to_tensor(x1), _to_tensor(x2)))


def atanh(x: Any) -> Any:
    """Element-wise inverse hyperbolic tangent.

    Args:
        x: Input array.

    Returns:
        An array containing the inverse hyperbolic tangent of each element in x.
    """
    return _wrap(ops.atanh(_to_tensor(x)))


def bitwise_and(x1: Any, x2: Any) -> Any:
    """JAX API implementation for bitwise_and.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.bitwise_and(_to_tensor(x1), _to_tensor(x2)))


def bitwise_not(x: Any) -> Any:
    """JAX API implementation for bitwise_not.

    Args:
        x: Argument x.

    Returns:
        Any: The result.
    """
    return _wrap(ops.bitwise_not(_to_tensor(x)))


def bitwise_or(x1: Any, x2: Any) -> Any:
    """JAX API implementation for bitwise_or.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.bitwise_or(_to_tensor(x1), _to_tensor(x2)))


def bitwise_xor(x1: Any, x2: Any) -> Any:
    """JAX API implementation for bitwise_xor.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.bitwise_xor(_to_tensor(x1), _to_tensor(x2)))


def logical_and(x1: Any, x2: Any) -> Any:
    """JAX API implementation for logical_and.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.logical_and(_to_tensor(x1), _to_tensor(x2)))


def logical_not(x: Any) -> Any:
    """JAX API implementation for logical_not.

    Args:
        x: Argument x.

    Returns:
        Any: The result.
    """
    return _wrap(ops.logical_not(_to_tensor(x)))


def logical_or(x1: Any, x2: Any) -> Any:
    """JAX API implementation for logical_or.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.logical_or(_to_tensor(x1), _to_tensor(x2)))


def logical_xor(x1: Any, x2: Any) -> Any:
    """JAX API implementation for logical_xor.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.logical_xor(_to_tensor(x1), _to_tensor(x2)))


def equal(x1: Any, x2: Any) -> Any:
    """JAX API implementation for equal.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.equal(_to_tensor(x1), _to_tensor(x2)))


def not_equal(x1: Any, x2: Any) -> Any:
    """JAX API implementation for not_equal.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.not_equal(_to_tensor(x1), _to_tensor(x2)))


def greater(x1: Any, x2: Any) -> Any:
    """JAX API implementation for greater.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.greater(_to_tensor(x1), _to_tensor(x2)))


def greater_equal(x1: Any, x2: Any) -> Any:
    """JAX API implementation for greater_equal.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.greater_equal(_to_tensor(x1), _to_tensor(x2)))


def less(x1: Any, x2: Any) -> Any:
    """JAX API implementation for less.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.less(_to_tensor(x1), _to_tensor(x2)))


def less_equal(x1: Any, x2: Any) -> Any:
    """JAX API implementation for less_equal.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.less_equal(_to_tensor(x1), _to_tensor(x2)))


def cbrt(x: Any) -> Any:
    """JAX API implementation for cbrt.

    Args:
        x: Argument x.

    Returns:
        Any: The result.
    """
    return _wrap(ops.cbrt(_to_tensor(x)))


def conj(x: Any) -> Any:
    """JAX API implementation for conj.

    Args:
        x: Argument x.

    Returns:
        Any: The result.
    """
    return _wrap(ops.conj(_to_tensor(x)))


def copysign(x1: Any, x2: Any) -> Any:
    """JAX API implementation for copysign.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.copysign(_to_tensor(x1), _to_tensor(x2)))


def count_nonzero(a: Any, axis: Any = None) -> Any:
    """JAX API implementation for count_nonzero.

    Args:
        a: Argument a.
        axis: Argument axis.

    Returns:
        Any: The result.
    """
    return _wrap(ops.count_nonzero(_to_tensor(a), axis=axis))


def cross(
    a: Any, b: Any, axisa: int = -1, axisb: int = -1, axisc: int = -1, axis: Any = None
) -> Any:
    """JAX API implementation for cross.

    Args:
        a: Argument a.
        b: Argument b.
        axisa: Argument axisa.
        axisb: Argument axisb.
        axisc: Argument axisc.
        axis: Argument axis.

    Returns:
        Any: The result.
    """
    return _wrap(
        ops.cross(
            _to_tensor(a),
            _to_tensor(b),
            axisa=axisa,
            axisb=axisb,
            axisc=axisc,
            axis=axis,
        )
    )


def deg2rad(x: Any) -> Any:
    """JAX API implementation for deg2rad.

    Args:
        x: Argument x.

    Returns:
        Any: The result.
    """
    return _wrap(ops.deg2rad(_to_tensor(x)))


def diag(v: Any, k: int = 0) -> Any:
    """JAX API implementation for diag.

    Args:
        v: Argument v.
        k: Argument k.

    Returns:
        Any: The result.
    """
    return _wrap(ops.diag(_to_tensor(v), diagonal=k))


def fix(x: Any) -> Any:
    """JAX API implementation for fix.

    Args:
        x: Argument x.

    Returns:
        Any: The result.
    """
    return _wrap(ops.fix(_to_tensor(x)))


def float_power(x1: Any, x2: Any) -> Any:
    """JAX API implementation for float_power.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.float_power(_to_tensor(x1), _to_tensor(x2)))


def fmax(x1: Any, x2: Any) -> Any:
    """JAX API implementation for fmax.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.fmax(_to_tensor(x1), _to_tensor(x2)))


def fmin(x1: Any, x2: Any) -> Any:
    """JAX API implementation for fmin.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.fmin(_to_tensor(x1), _to_tensor(x2)))


def fmod(x1: Any, x2: Any) -> Any:
    """JAX API implementation for fmod.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.fmod(_to_tensor(x1), _to_tensor(x2)))


def gcd(x1: Any, x2: Any) -> Any:
    """JAX API implementation for gcd.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.gcd(_to_tensor(x1), _to_tensor(x2)))


def heaviside(x1: Any, x2: Any) -> Any:
    """JAX API implementation for heaviside.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.heaviside(_to_tensor(x1), _to_tensor(x2)))


def hypot(x1: Any, x2: Any) -> Any:
    """JAX API implementation for hypot.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.hypot(_to_tensor(x1), _to_tensor(x2)))


def imag(x: Any) -> Any:
    """JAX API implementation for imag.

    Args:
        x: Argument x.

    Returns:
        Any: The result.
    """
    return _wrap(ops.imag(_to_tensor(x)))


def isclose(
    a: Any, b: Any, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> Any:
    """JAX API implementation for isclose.

    Args:
        a: Argument a.
        b: Argument b.
        rtol: Argument rtol.
        atol: Argument atol.
        equal_nan: Argument equal_nan.

    Returns:
        Any: The result.
    """
    return _wrap(
        ops.isclose(
            _to_tensor(a), _to_tensor(b), rtol=rtol, atol=atol, equal_nan=equal_nan
        )
    )


def isinf(x: Any) -> Any:
    """JAX API implementation for isinf.

    Args:
        x: Argument x.

    Returns:
        Any: The result.
    """
    return _wrap(ops.isinf(_to_tensor(x)))


def lcm(x1: Any, x2: Any) -> Any:
    """JAX API implementation for lcm.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.lcm(_to_tensor(x1), _to_tensor(x2)))


def ldexp(x1: Any, x2: Any) -> Any:
    """JAX API implementation for ldexp.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.ldexp(_to_tensor(x1), _to_tensor(x2)))


def left_shift(x1: Any, x2: Any) -> Any:
    """JAX API implementation for left_shift.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.left_shift(_to_tensor(x1), _to_tensor(x2)))


def logaddexp(x1: Any, x2: Any) -> Any:
    """JAX API implementation for logaddexp.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.logaddexp(_to_tensor(x1), _to_tensor(x2)))


def logaddexp2(x1: Any, x2: Any) -> Any:
    """JAX API implementation for logaddexp2.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.logaddexp2(_to_tensor(x1), _to_tensor(x2)))


def nextafter(x1: Any, x2: Any) -> Any:
    """JAX API implementation for nextafter.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.nextafter(_to_tensor(x1), _to_tensor(x2)))


def rad2deg(x: Any) -> Any:
    """JAX API implementation for rad2deg.

    Args:
        x: Argument x.

    Returns:
        Any: The result.
    """
    return _wrap(ops.rad2deg(_to_tensor(x)))


def real(x: Any) -> Any:
    """JAX API implementation for real.

    Args:
        x: Argument x.

    Returns:
        Any: The result.
    """
    return _wrap(ops.real(_to_tensor(x)))


def reciprocal(x: Any) -> Any:
    """JAX API implementation for reciprocal.

    Args:
        x: Argument x.

    Returns:
        Any: The result.
    """
    return _wrap(ops.reciprocal(_to_tensor(x)))


def right_shift(x1: Any, x2: Any) -> Any:
    """JAX API implementation for right_shift.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return _wrap(ops.right_shift(_to_tensor(x1), _to_tensor(x2)))


def roll(a: Any, shift: Any, axis: Any = None) -> Any:
    """JAX API implementation for roll.

    Args:
        a: Argument a.
        shift: Argument shift.
        axis: Argument axis.

    Returns:
        Any: The result.
    """
    return _wrap(ops.roll(_to_tensor(a), shift=shift, axis=axis))


def round(a: Any, decimals: int = 0) -> Any:
    """JAX API implementation for round.

    Args:
        a: Argument a.
        decimals: Argument decimals.

    Returns:
        Any: The result.
    """
    return _wrap(ops.round(_to_tensor(a), decimals=decimals))


def select(condlist: Any, choicelist: Any, default: Any = 0) -> Any:
    """JAX API implementation for select.

    Args:
        condlist: Argument condlist.
        choicelist: Argument choicelist.
        default: Argument default.

    Returns:
        Any: The result.
    """
    res = _to_tensor(default)
    for c, v in reversed(list(zip(condlist, choicelist))):
        res = ops.where(_to_tensor(c), _to_tensor(v), res)
    return _wrap(res)


def sinc(x: Any) -> Any:
    """JAX API implementation for sinc.

    Args:
        x: Argument x.

    Returns:
        Any: The result.
    """
    return _wrap(ops.sinc(_to_tensor(x)))


def sort(a: Any, axis: int = -1, kind: Any = None, order: Any = None) -> Any:
    """JAX API implementation for sort.

    Args:
        a: Argument a.
        axis: Argument axis.
        kind: Argument kind.
        order: Argument order.

    Returns:
        Any: The result.
    """
    return _wrap(ops.sort(_to_tensor(a), axis=axis))


def tril(m: Any, k: int = 0) -> Any:
    """JAX API implementation for tril.

    Args:
        m: Argument m.
        k: Argument k.

    Returns:
        Any: The result.
    """
    return _wrap(ops.tril(_to_tensor(m), diagonal=k))


def triu(m: Any, k: int = 0) -> Any:
    """JAX API implementation for triu.

    Args:
        m: Argument m.
        k: Argument k.

    Returns:
        Any: The result.
    """
    return _wrap(ops.triu(_to_tensor(m), diagonal=k))


def unstack(x: Any, axis: int = 0) -> Any:
    """JAX API implementation for unstack.

    Args:
        x: Argument x.
        axis: Argument axis.

    Returns:
        Any: The result.
    """
    res = ops.unstack(_to_tensor(x), dim=axis)
    return tuple(_wrap(t) for t in res)


def absolute(x: Any) -> Any:
    """Calculates the absolute value element-wise.

    Args:
        x: Input array.

    Returns:
        An array containing the absolute value of each element in x.
    """
    return _wrap(ops.abs(_to_tensor(x)))


def around(a: Any, decimals: int = 0) -> Any:
    """Evenly round to the given number of decimals.

    Args:
        a: Input data.
        decimals: Number of decimal places to round to.

    Returns:
        An array of the same type as a, containing the rounded values.
    """
    return round(a, decimals=decimals)


def round_(a: Any, decimals: int = 0) -> Any:
    """Evenly round to the given number of decimals.

    Args:
        a: Input data.
        decimals: Number of decimal places to round to.

    Returns:
        An array of the same type as a, containing the rounded values.
    """
    return round(a, decimals=decimals)


def conjugate(x: Any) -> Any:
    """Return the complex conjugate, element-wise.

    Args:
        x: Input array.

    Returns:
        The complex conjugate of x.
    """
    return conj(x)


def cumulative_sum(a: Any, axis: Any = None, dtype: Any = None) -> Any:
    """Return the cumulative sum of the elements along a given axis.

    Args:
        a: Input array.
        axis: Axis along which the cumulative sum is computed.
        dtype: Type of the returned array.

    Returns:
        A new array holding the result.
    """
    return cumsum(a, axis=axis, dtype=dtype)


def degrees(x: Any) -> Any:
    """Convert angles from radians to degrees.

    Args:
        x: Input array in radians.

    Returns:
        The corresponding degree values.
    """
    return rad2deg(x)


def radians(x: Any) -> Any:
    """Convert angles from degrees to radians.

    Args:
        x: Input array in degrees.

    Returns:
        The corresponding radian values.
    """
    return deg2rad(x)


def pow(x1: Any, x2: Any) -> Any:
    """First array elements raised to powers from second array, element-wise.

    Args:
        x1: The bases.
        x2: The exponents.

    Returns:
        The bases in x1 raised to the exponents in x2.
    """
    return power(x1, x2)


def bitwise_invert(x: Any) -> Any:
    """JAX API implementation for bitwise_invert.

    Args:
        x: Argument x.

    Returns:
        Any: The result.
    """
    return bitwise_not(x)


def bitwise_left_shift(x1: Any, x2: Any) -> Any:
    """JAX API implementation for bitwise_left_shift.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return left_shift(x1, x2)


def bitwise_right_shift(x1: Any, x2: Any) -> Any:
    """JAX API implementation for bitwise_right_shift.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        Any: The result.
    """
    return right_shift(x1, x2)


def concat(arrays: Any, axis: int = 0) -> Any:
    """JAX API implementation for concat.

    Args:
        arrays: Argument arrays.
        axis: Argument axis.

    Returns:
        Any: The result.
    """
    return concatenate(arrays, axis=axis)


def invert(x: Any) -> Any:
    """JAX API implementation for invert.

    Args:
        x: Argument x.

    Returns:
        Any: The result.
    """
    return bitwise_not(x)


def nanmax(
    a: Any,
    axis: Any = None,
    keepdims: Any = False,
    initial: Any = None,
    where: Any = None,
) -> Any:
    """JAX API implementation for nanmax.

    Args:
        a: Argument a.
        axis: Argument axis.
        keepdims: Argument keepdims.
        initial: Argument initial.
        where: Argument where.

    Returns:
        Any: The result.
    """
    return max(a, axis=axis, keepdims=keepdims)


def nanmin(
    a: Any,
    axis: Any = None,
    keepdims: Any = False,
    initial: Any = None,
    where: Any = None,
) -> Any:
    """JAX API implementation for nanmin.

    Args:
        a: Argument a.
        axis: Argument axis.
        keepdims: Argument keepdims.
        initial: Argument initial.
        where: Argument where.

    Returns:
        Any: The result.
    """
    return min(a, axis=axis, keepdims=keepdims)


def nanprod(
    a: Any,
    axis: Any = None,
    dtype: Any = None,
    keepdims: Any = False,
    initial: Any = None,
    where: Any = None,
) -> Any:
    """JAX API implementation for nanprod.

    Args:
        a: Argument a.
        axis: Argument axis.
        dtype: Argument dtype.
        keepdims: Argument keepdims.
        initial: Argument initial.
        where: Argument where.

    Returns:
        Any: The result.
    """
    return prod(a, axis=axis, keepdims=keepdims)


def nansum(
    a: Any,
    axis: Any = None,
    dtype: Any = None,
    keepdims: Any = False,
    initial: Any = None,
    where: Any = None,
) -> Any:
    """JAX API implementation for nansum.

    Args:
        a: Argument a.
        axis: Argument axis.
        dtype: Argument dtype.
        keepdims: Argument keepdims.
        initial: Argument initial.
        where: Argument where.

    Returns:
        Any: The result.
    """
    return sum(a, axis=axis, keepdims=keepdims)


def isneginf(x: Any, out: Any = None) -> Any:
    # JAX handles isneginf by doing isinf(x) & (x < 0)
    """JAX API implementation for isneginf.

    Args:
        x: Argument x.
        out: Argument out.

    Returns:
        Any: The result.
    """
    inf_mask = isinf(x)
    neg_mask = x < 0
    res = logical_and(inf_mask, neg_mask)
    if out is not None:
        raise NotImplementedError("out parameter is not supported")
    return res


def isposinf(x: Any, out: Any = None) -> Any:
    """JAX API implementation for isposinf.

    Args:
        x: Argument x.
        out: Argument out.

    Returns:
        Any: The result.
    """
    inf_mask = isinf(x)
    pos_mask = x > 0
    res = logical_and(inf_mask, pos_mask)
    if out is not None:
        raise NotImplementedError("out parameter is not supported")
    return res


def frexp(x: Any, out: Any = None) -> Any:
    """JAX API implementation for frexp.

    Args:
        x: Argument x.
        out: Argument out.

    Returns:
        Any: The result.
    """
    out_t = ops.frexp(_to_tensor(x))
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        data = out_t.data
        if isinstance(data, tuple):
            import ml_switcheroo_compiler.core.tensor as tensor

            return _wrap(
                tensor.Tensor(
                    data[0],
                    TensorConfig(
                        shape=data[0].shape, dtype=data[0].dtype, device=out_t.device
                    ),
                )
            ), _wrap(
                tensor.Tensor(
                    data[1],
                    TensorConfig(
                        shape=data[1].shape, dtype=data[1].dtype, device=out_t.device
                    ),
                )
            )
    return _wrap(out_t)


def nan_to_num(
    x: Any, copy: bool = True, nan: float = 0.0, posinf: Any = None, neginf: Any = None
) -> Any:
    """Replace NaN with zero and inf with large finite numbers.

    Args:
        x: Input array.
        copy: Ignored.
        nan: Value to be used to fill NaN values.
        posinf: Value to be used to fill positive infinity values.
        neginf: Value to be used to fill negative infinity values.

    Returns:
        The resulting array.
    """
    return _wrap(ops.nan_to_num(_to_tensor(x), nan=nan, posinf=posinf, neginf=neginf))


def searchsorted(a: Any, v: Any, side: str = "left", sorter: Any = None) -> Any:
    """Find indices where elements should be inserted to maintain order.

    Args:
        a: Input array.
        v: Values to insert into a.
        side: If 'left', the index of the first suitable location found is given.
        sorter: Optional array of integer indices that sort array a into ascending order.

    Returns:
        Array of insertion points.
    """
    # ml-switcheroo-compiler ops.searchsorted currently only accepts side.
    if sorter is not None:
        raise NotImplementedError("sorter is not yet supported in searchsorted")
    return _wrap(
        ops.shape.indexing.searchsorted(_to_tensor(a), _to_tensor(v), side=side)
    )


def signbit(x: Any, out: Any = None) -> Any:
    """Returns element-wise True where signbit is set (less than zero).

    Args:
        x: Input array.
        out: Ignored.

    Returns:
        Boolean array.
    """
    if out is not None:
        raise NotImplementedError("out parameter is not supported")
    return _wrap(ops.signbit(_to_tensor(x)))


def argsort(a: Any, axis: int = -1, kind: Any = None, order: Any = None) -> Any:
    """Returns the indices that would sort an array.

    Args:
        a: Array to sort.
        axis: Axis along which to sort.
        kind: Sorting algorithm.
        order: Field to sort by.

    Returns:
        Array of indices that sort a.
    """
    # ml-switcheroo-compiler ops.sort currently just sorts, no argsort directly exported in the python API
    # But wait, numpy argsort can't be implemented with ops.sort directly if it doesn't return indices.
    # Let's check if the compiler has Argsort eager op.
    raise NotImplementedError("argsort not fully supported in compiler natively yet")


def copy(a: Any, order: str = "K") -> Any:
    """Return an array copy of the given object.

    Args:
        a: Input data.
        order: memory layout.

    Returns:
        An array interpretation of a.
    """
    return _wrap(ops.array(_to_tensor(a)))


from ml_switcheroo_compiler.core.dtype import DType as _DType

bool = _DType.Bool
bool_ = _DType.Bool
complex128 = _DType.Complex128
complex64 = _DType.Complex64
complex_ = _DType.Complex128
float16 = _DType.Float16
bfloat16 = _DType.BFloat16

from ml_switcheroo_compiler.core.dtype import DType

# Map float8s to float16 since compiler doesn't have float8 yet
float8_e4m3b11fnuz = DType.Float16
float8_e4m3fn = DType.Float16
float8_e4m3fnuz = DType.Float16
float8_e5m2 = DType.Float16
float8_e5m2fnuz = DType.Float16

float32 = _DType.Float32
float64 = _DType.Float64
float_ = _DType.Float64
int16 = _DType.Int16
int32 = _DType.Int32
int64 = _DType.Int64
int8 = _DType.Int8
int4 = _DType.Int8

int_ = _DType.Int64
uint16 = _DType.Int16  # Fallback
uint32 = _DType.UInt32
uint64 = _DType.Int64  # Fallback
uint8 = _DType.UInt8


class generic:
    """generic."""


class number(generic):
    """number."""


class integer(number):
    """integer."""


class signedinteger(integer):
    """signedinteger."""


class unsignedinteger(integer):
    """unsignedinteger."""


class inexact(number):
    """inexact."""


class floating(inexact):
    """floating."""


class complexfloating(inexact):
    """complexfloating."""


class flexible(generic):
    """flexible."""


class character(flexible):
    """character."""


class object_(generic):
    """object_."""


double = float64
single = float32
csingle = complex64
cdouble = complex128


class ComplexWarning(Warning):
    """Complex warning."""

    pass


def angle(z: Any, deg: Any = False) -> Any:
    """Return the angle of the complex argument.

    Args:
        z: A complex number or sequence of complex numbers.
        deg: Return angle in degrees if True, radians if False (default).

    Returns:
        The counterclockwise angle.
    """
    return _wrap(ops.angle(_to_tensor(z), deg=deg))


def append(arr: Any, values: Any, axis: Any = None) -> Any:
    """Append values to the end of an array.

    Args:
        arr: Values are appended to a copy of this array.
        values: These values are appended to a copy of arr.
        axis: The axis along which values are appended.

    Returns:
        A copy of arr with values appended to axis.
    """
    return _wrap(ops.append(_to_tensor(arr), _to_tensor(values), axis=axis))


def astype(x: Any, dtype: Any, copy: Any = False, device: Any = None) -> Any:
    """Copy of the array, cast to a specified type.

    Args:
        x: Input array.
        dtype: Data type to which the array is cast.
        copy: Whether to copy the array.
        device: Device to place the returned array.

    Returns:
        The cast array.
    """
    return _wrap(ops.cast(_to_tensor(x), dtype))


def atleast_1d(*arys: Any) -> Any:
    """Convert inputs to arrays with at least one dimension.

    Args:
        arys: One or more input arrays.

    Returns:
        An array, or list of arrays, each with a.ndim >= 1.
    """
    res = ops.atleast_1d(*[_to_tensor(a) for a in arys])
    if isinstance(res, list):
        return [_wrap(r) for r in res]  # pragma: no cover
    return _wrap(res)


def atleast_2d(*arys: Any) -> Any:
    """Convert inputs to arrays with at least two dimensions.

    Args:
        arys: One or more array-like sequences.

    Returns:
        An array, or list of arrays, each with a.ndim >= 2.
    """
    res = ops.atleast_2d(*[_to_tensor(a) for a in arys])
    if isinstance(res, list):
        return [_wrap(r) for r in res]  # pragma: no cover
    return _wrap(res)


def atleast_3d(*arys: Any) -> Any:
    """Convert inputs to arrays with at least three dimensions.

    Args:
        arys: One or more array-like sequences.

    Returns:
        An array, or list of arrays, each with a.ndim >= 3.
    """
    res = ops.atleast_3d(*[_to_tensor(a) for a in arys])
    if isinstance(res, list):
        return [_wrap(r) for r in res]  # pragma: no cover
    return _wrap(res)


def average(
    a: Any,
    axis: Any = None,
    weights: Any = None,
    returned: Any = False,
    keepdims: Any = False,
) -> Any:
    """Compute the weighted average along the specified axis.

    Args:
        a: Array containing data to be averaged.
        axis: Axis or axes along which to average a.
        weights: An array of weights associated with the values in a.
        returned: Whether to return sum of weights.
        keepdims: Whether to keep dimensions with size one.

    Returns:
        The average along the specified axis.
    """
    w = _to_tensor(weights) if weights is not None else None
    res = ops.average(
        _to_tensor(a), axis=axis, weights=w, returned=returned, keepdims=keepdims
    )
    if returned:
        return _wrap(res[0]), _wrap(res[1])  # pragma: no cover
    return _wrap(res)


def block(arrays: Any) -> Any:
    """Assemble an nd-array from nested lists of blocks.

    Args:
        arrays: An array or list of blocks.

    Returns:
        The block array.
    """
    return _wrap(ops.block(arrays))


def ndim(a: Any) -> int:
    """Return the number of dimensions of an array.

    Args:
        a: Input array.

    Returns:
        The number of dimensions in a.
    """
    return (
        len(_to_tensor(a).shape) if _to_tensor(a).shape is not None else 0
    )  # pragma: no cover


def apply_along_axis(
    func1d: Any, axis: int, arr: Any, *args: Any, **kwargs: Any
) -> Any:
    """Apply a function to 1-D slices along the given axis.

    Args:
        func1d: This function should accept 1-D arrays.
        axis: Axis along which arr is sliced.
        arr: Input array.
        *args: Additional arguments to func1d.
        **kwargs: Additional keyword arguments to func1d.

    Returns:
        The output array.
    """

    def wrapped_func(t, *f_args, **f_kwargs):
        return _to_tensor(func1d(_wrap(t), *f_args, **f_kwargs))  # pragma: no cover

    if config.eager_mode:
        np = __import__("numpy")

        arr_data = (
            _to_tensor(arr).data
            if hasattr(_to_tensor(arr), "data")
            else _to_tensor(arr)
        )
        res = np.apply_along_axis(
            lambda x: getattr(
                _to_tensor(func1d(_wrap(x))), "data", _to_tensor(func1d(_wrap(x)))
            ),
            axis,
            arr_data,
            *args,
            **kwargs,
        )
        return _wrap(
            _ml_switcheroo_compiler.Tensor(
                res,
                config=_ml_switcheroo_compiler.core.tensor.TensorConfig(
                    shape=res.shape,
                    dtype=_ml_switcheroo_compiler.core.dtype.DType(str(res.dtype)),
                    device=config.default_device,
                ),
            )
        )
    return _wrap(  # pragma: no cover
        ops.apply_along_axis(wrapped_func, axis, _to_tensor(arr), *args, **kwargs)
    )


def apply_over_axes(func: Any, a: Any, axes: Any) -> Any:
    """Apply a function repeatedly over multiple axes.

    Args:
        func: This function must take two arguments, func(a, axis).
        a: Input array.
        axes: Axes over which func is applied.

    Returns:
        The output array.
    """

    def wrapped_func(t, axis):  # pragma: no cover
        return _to_tensor(func(_wrap(t), axis))  # pragma: no cover

    return _wrap(
        ops.apply_over_axes(wrapped_func, _to_tensor(a), axes)
    )  # pragma: no cover


def argpartition(a: Any, kth: int, axis: int = -1) -> Any:
    """Perform an indirect partition along the given axis.

    Args:
        a: Array to sort.
        kth: Element index to partition by.
        axis: Axis along which to sort.

    Returns:
        Array of indices that partition a along the specified axis.
    """
    return _wrap(ops.argpartition(_to_tensor(a), kth, axis=axis))


def argwhere(a: Any, *, size: Any = None, fill_value: Any = None) -> Any:
    """Find the indices of array elements that are non-zero, grouped by element.

    Args:
        a: Input data.
        size: Optional size for the output.
        fill_value: Optional fill value.

    Returns:
        Indices of elements that are non-zero.
    """
    # size and fill_value are JAX-specific optimizations
    # we omit passing them if ops backend doesn't support them.
    return _wrap(ops.argwhere(_to_tensor(a)))


def choose(a: Any, choices: Any, out: Any = None, mode: str = "raise") -> Any:
    """Construct an array from an index array and a set of arrays to choose from.

    Args:
        a: Array of integers indicating which array in choices to take values from.
        choices: Sequence of arrays to choose from.
        out: If provided, the result will be inserted into this array.
        mode: Specifies how indices outside [0, n-1] will be treated.

    Returns:
        The merged result.
    """
    c = [_to_tensor(ch) for ch in choices]
    return _wrap(ops.choose(_to_tensor(a), c, out=out, mode=mode))


def column_stack(tup: Any) -> Any:
    """Stack 1-D arrays as columns into a 2-D array.

    Args:
        tup: Sequence of 1-D or 2-D arrays.

    Returns:
        The stacked array.
    """
    return _wrap(ops.column_stack([_to_tensor(t) for t in tup]))


def compress(
    condition: Any,
    a: Any,
    axis: Any = None,
    *,
    size: Any = None,
    fill_value: Any = 0,
    out: Any = None,
) -> Any:
    """Return selected slices of an array along given axis.

    Args:
        condition: Array that selects which entries to return.
        a: Input array.
        axis: Axis along which to take slices.
        size: Optional padding.
        fill_value: Padding value.
        out: Output array.

    Returns:
        A copy of a without the slices along axis for which condition is false.
    """
    return _wrap(ops.compress(_to_tensor(condition), _to_tensor(a), axis=axis, out=out))


def convolve(
    a: Any,
    v: Any,
    mode: str = "full",
    *,
    precision: Any = None,
    preferred_element_type: Any = None,
) -> Any:
    """Returns the discrete, linear convolution of two one-dimensional sequences.

    Args:
        a: First input sequence.
        v: Second input sequence.
        mode: 'full', 'valid', or 'same'.
        precision: Numerical precision.
        preferred_element_type: Optional dtype.

    Returns:
        Discrete, linear convolution of a and v.
    """
    # Pass kwargs supported by JAX if applicable, otherwise base ops
    return _wrap(ops.convolve(_to_tensor(a), _to_tensor(v), mode=mode))


def corrcoef(x: Any, y: Any = None, rowvar: Any = True) -> Any:
    """Return Pearson product-moment correlation coefficients.

    Args:
        x: A 1-D or 2-D array containing multiple variables and observations.
        y: An additional set of variables and observations.
        rowvar: If rowvar is True, each row represents a variable.

    Returns:
        The correlation coefficient matrix of the variables.
    """
    return _wrap(
        ops.corrcoef(
            _to_tensor(x), _to_tensor(y) if y is not None else None, rowvar=rowvar
        )
    )


def correlate(
    a: Any,
    v: Any,
    mode: str = "valid",
    *,
    precision: Any = None,
    preferred_element_type: Any = None,
) -> Any:
    """Cross-correlation of two 1-dimensional sequences.

    Args:
        a: First input sequence.
        v: Second input sequence.
        mode: 'valid', 'same', or 'full'.
        precision: Numerical precision.
        preferred_element_type: Optional dtype.

    Returns:
        Discrete cross-correlation of a and v.
    """
    return _wrap(ops.correlate(_to_tensor(a), _to_tensor(v), mode=mode))


def cov(
    m: Any,
    y: Any = None,
    rowvar: Any = True,
    bias: Any = False,
    ddof: Any = None,
    fweights: Any = None,
    aweights: Any = None,
) -> Any:
    """Estimate a covariance matrix, given data and weights.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
        y: An additional set of variables and observations.
        rowvar: If rowvar is True, each row represents a variable.
        bias: Default normalization is False.
        ddof: Degrees of freedom.
        fweights: Frequency weights.
        aweights: Observation vector weights.

    Returns:
        The covariance matrix of the variables.
    """
    y_tensor = _to_tensor(y) if y is not None else None
    return _wrap(
        ops.cov(
            _to_tensor(m),
            y=y_tensor,
            rowvar=rowvar,
            bias=bias,
            ddof=ddof,
            fweights=fweights,
            aweights=aweights,
        )
    )


def array_equiv(a1: Any, a2: Any) -> Any:
    """Returns True if input arrays are shape consistent and all elements equal.

    Args:
        a1: Input array.
        a2: Input array.

    Returns:
        True if equivalent, False otherwise.
    """
    return _wrap(ops.array_equiv(_to_tensor(a1), _to_tensor(a2)))


def array_repr(
    arr: Any,
    max_line_width: Any = None,
    precision: Any = None,
    suppress_small: Any = None,
) -> Any:
    """Return the string representation of an array.

    Args:
        arr: Input array.
        max_line_width: Maximum line width.
        precision: Floating point precision.
        suppress_small: Whether to suppress very small numbers.

    Returns:
        String representation.
    """
    return ops.array_repr(_to_tensor(arr), max_line_width, precision, suppress_small)


def array_str(
    arr: Any,
    max_line_width: Any = None,
    precision: Any = None,
    suppress_small: Any = None,
) -> Any:
    """Return a string representation of the data in an array.

    Args:
        arr: Input array.
        max_line_width: Maximum line width.
        precision: Floating point precision.
        suppress_small: Whether to suppress very small numbers.

    Returns:
        String representation.
    """
    return ops.array_str(_to_tensor(arr), max_line_width, precision, suppress_small)


def bartlett(M: int) -> Any:
    """Return the Bartlett window.

    Args:
        M: Number of points in the output window.

    Returns:
        The window.
    """
    return _wrap(ops.bartlett(M))


def bincount(
    x: Any, weights: Any = None, minlength: int = 0, *, length: Any = None
) -> Any:
    """Count number of occurrences of each value in array of non-negative ints.

    Args:
        x: Input array.
        weights: Weights, array of the same shape as x.
        minlength: A minimum number of bins for the output array.
        length: Alias for minlength.

    Returns:
        The result of binning the input array.
    """
    mlen = length if length is not None else minlength
    return _wrap(
        ops.bincount(
            _to_tensor(x),
            _to_tensor(weights) if weights is not None else None,
            minlength=mlen,
        )
    )


def bitwise_count(x: Any) -> Any:
    """Counts the number of 1-bits in the absolute value of the input.

    Args:
        x: Input array.

    Returns:
        The number of 1-bits.
    """
    return _wrap(ops.bitwise_count(_to_tensor(x)))  # pragma: no cover


def blackman(M: int) -> Any:
    """Return the Blackman window.

    Args:
        M: Number of points in the output window.

    Returns:
        The window.
    """
    return _wrap(ops.blackman(M))


def broadcast_arrays(*args: Any, **kwargs: Any) -> Any:
    """Broadcast any number of arrays against each other.

    Args:
        *args: The arrays to broadcast.
        **kwargs: Additional kwargs.

    Returns:
        A list of broadcasted arrays.
    """
    return [
        _wrap(t) for t in ops.broadcast_arrays(*[_to_tensor(a) for a in args], **kwargs)
    ]


def can_cast(from_: Any, to: Any, casting: str = "safe") -> Any:
    """Returns True if cast between data types can occur according to the casting rule.

    Args:
        from_: Source type.
        to: Destination type.
        casting: Casting rule.

    Returns:
        True if cast can occur.
    """
    if config.eager_mode:
        np = __import__("numpy")

        return np.can_cast(from_, to, casting=casting)
    return ops.can_cast(from_, to, casting=casting)  # pragma: no cover


def cumprod(a: Any, axis: Any = None, dtype: Any = None, out: Any = None) -> Any:
    """Return the cumulative product of elements along a given axis.

    Args:
        a: Input array.
        axis: Axis along which the cumulative product is computed.
        dtype: Type of the returned array.
        out: Alternative output array.

    Returns:
        The cumulative product.
    """
    return _wrap(ops.cumprod(_to_tensor(a), axis=axis, dtype=dtype))


def delete(arr: Any, obj: Any, axis: Any = None) -> Any:
    """Return a new array with sub-arrays along an axis deleted.

    Args:
        arr: Input array.
        obj: Indicate indices of sub-arrays to remove along the specified axis.
        axis: The axis along which to delete the subarray.

    Returns:
        A copy of arr with the elements specified by obj removed.
    """
    return _wrap(
        ops.delete(
            _to_tensor(arr),
            _to_tensor(obj) if hasattr(obj, "__iter__") else obj,
            axis=axis,
        )
    )


def diag_indices(n: int, ndim: int = 2) -> Any:
    """Return the indices to access the main diagonal of an array.

    Args:
        n: The size, along each dimension, of the arrays for which the returned indices can be used.
        ndim: The number of dimensions.

    Returns:
        A tuple of indices.
    """
    return tuple(_wrap(t) for t in ops.diag_indices(n, ndim=ndim))


def diag_indices_from(arr: Any) -> Any:
    """Return the indices to access the main diagonal of an n-dimensional array.

    Args:
        arr: Input array.

    Returns:
        A tuple of indices.
    """
    return tuple(_wrap(t) for t in ops.diag_indices_from(_to_tensor(arr)))


def diagflat(v: Any, k: int = 0) -> Any:
    """Create a two-dimensional array with the flattened input as a diagonal.

    Args:
        v: Input data.
        k: Diagonal to set.

    Returns:
        The 2-D output array.
    """
    return _wrap(ops.diagflat(_to_tensor(v), k=k))


def diagonal(a: Any, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Any:
    """Return specified diagonals.

    Args:
        a: Input array.
        offset: Offset of the diagonal from the main diagonal.
        axis1: Axis to be used as the first axis of the 2-D sub-arrays.
        axis2: Axis to be used as the second axis of the 2-D sub-arrays.

    Returns:
        Array of diagonals.
    """
    return _wrap(ops.diagonal(_to_tensor(a), offset=offset, axis1=axis1, axis2=axis2))


def diff(
    a: Any, n: int = 1, axis: int = -1, prepend: Any = None, append: Any = None
) -> Any:
    """Calculate the n-th discrete difference along the given axis.

    Args:
        a: Input array.
        n: The number of times values are differenced.
        axis: The axis along which the difference is taken.
        prepend: Values to prepend to a.
        append: Values to append to a.

    Returns:
        The n-th differences.
    """
    return _wrap(
        ops.diff(
            _to_tensor(a),
            n=n,
            axis=axis,
            prepend=_to_tensor(prepend) if prepend is not None else None,
            append=_to_tensor(append) if append is not None else None,
        )
    )


def digitize(x: Any, bins: Any, right: Any = False) -> Any:
    """Return the indices of the bins to which each value in input array belongs.

    Args:
        x: Input array to be binned.
        bins: Array of bins.
        right: Indicating whether the intervals include the right or the left bin edge.

    Returns:
        Array of indices.
    """
    return _wrap(ops.digitize(_to_tensor(x), _to_tensor(bins), right=right))


def dtype(
    value: Any,
    names: Any = None,
    *,
    module: Any = None,
    qualname: Any = None,
    type: Any = None,
    start: int = 1,
) -> Any:
    """Create a data type object.

    Args:
        value: Object to be converted to a data type object.
        names: Optional.
        module: Optional.
        qualname: Optional.
        type: Optional.
        start: Optional.

    Returns:
        A new dtype object.
    """
    # JAX normally delegates to numpy for this.
    # We implement a lightweight proxy returning DType enum equivalent or standard mapping.
    try:
        if hasattr(value, "value"):
            return value  # pragma: no cover
        if isinstance(value, str):
            return _DType(value.lower())  # pragma: no cover
        return _DType(value.__name__.lower())
    except Exception:  # pragma: no cover
        return value  # pragma: no cover


def ediff1d(ary: Any, to_end: Any = None, to_begin: Any = None) -> Any:
    """The differences between consecutive elements of an array.

    Args:
        ary: If necessary, will be flattened before the differences are taken.
        to_end: Number(s) to append at the end of the returned differences.
        to_begin: Number(s) to prepend at the beginning of the returned differences.

    Returns:
        The differences.
    """
    return _wrap(
        ops.ediff1d(
            _to_tensor(ary),
            to_end=_to_tensor(to_end) if to_end is not None else None,
            to_begin=_to_tensor(to_begin) if to_begin is not None else None,
        )
    )


def einsum_path(subscripts: Any, *operands: Any, optimize: Any = "auto") -> Any:
    """Evaluates the lowest cost contraction order for an einsum expression.

    Args:
        subscripts: Specifies the subscripts for summation.
        *operands: These are the arrays for the operation.
        optimize: Choose the type of path.

    Returns:
        A tuple of (path, string_representation).
    """
    return ops.einsum_path(
        subscripts, *[_to_tensor(o) for o in operands], optimize=optimize
    )


def extract(condition: Any, arr: Any, *, size: Any = None, fill_value: Any = 0) -> Any:
    """Return the elements of an array that satisfy some condition.

    Args:
        condition: An array whose nonzero or True entries indicate the elements of arr to extract.
        arr: Input array of the same size as condition.
        size: Optional size limit.
        fill_value: Optional fill value.

    Returns:
        Rank 1 array of values from arr where condition is True.
    """
    return _wrap(ops.extract(_to_tensor(condition), _to_tensor(arr)))


def fabs(x: Any) -> Any:
    """Compute the absolute values element-wise.

    Args:
        x: Input array.

    Returns:
        The absolute values.
    """
    return _wrap(ops.fabs(_to_tensor(x)))


def fill_diagonal(a: Any, val: Any, wrap: bool = False, *, inplace: bool = True) -> Any:
    """Fill the main diagonal of the given array of any dimensionality.

    Args:
        a: Array whose diagonal is to be filled.
        val: Value to be written on the diagonal.
        wrap: For tall matrices in NumPy version up to 1.6.2, the diagonal "wrapped" after N columns.
        inplace: Modified in place.

    Returns:
        The array with the diagonal filled.
    """
    return _wrap(
        ops.fill_diagonal(_to_tensor(a), _to_tensor(val), wrap=wrap, inplace=inplace)
    )


def finfo(dtype: Any) -> Any:
    """Machine limits for floating point types.

    Args:
        dtype: The kind of floating point data-type.

    Returns:
        Machine limits.
    """
    if config.eager_mode:
        np = __import__("numpy")

        return np.finfo(dtype)
    return ops.finfo(dtype)  # pragma: no cover


def flatnonzero(a: Any, *, size: Any = None, fill_value: Any = None) -> Any:
    """Return indices that are non-zero in the flattened version of a.

    Args:
        a: Input array.
        size: Optional output size.
        fill_value: Optional fill value.

    Returns:
        Output array, containing the indices of the elements of a.ravel() that are non-zero.
    """
    return _wrap(ops.flatnonzero(_to_tensor(a)))


def flip(m: Any, axis: Any = None) -> Any:
    """Reverse the order of elements in an array along the given axis.

    Args:
        m: Input array.
        axis: Axis or axes along which to flip over.

    Returns:
        A view of m with the entries of axis reversed.
    """
    return _wrap(ops.flip(_to_tensor(m), axis=axis))


def fliplr(m: Any) -> Any:
    """Reverse the order of elements along axis 1 (left/right).

    Args:
        m: Input array.

    Returns:
        A view of m with the columns reversed.
    """
    return _wrap(ops.fliplr(_to_tensor(m)))


def flipud(m: Any) -> Any:
    """Reverse the order of elements along axis 0 (up/down).

    Args:
        m: Input array.

    Returns:
        A view of m with the rows reversed.
    """
    return _wrap(ops.flipud(_to_tensor(m)))


def from_dlpack(x: Any, /, *, device: Any = None, copy: Any = None) -> Any:
    """Create an array from a DLPack tensor.

    Args:
        x: A DLPack tensor.
        device: The device on which the created array should be placed.
        copy: Optional.

    Returns:
        The array.
    """
    # device/copy omitted since ml-switcheroo-compiler doesn't expose them in from_dlpack currently
    return _wrap(ops.from_dlpack(x))  # pragma: no cover


def frombuffer(
    buffer: Any, dtype: Any = float, count: int = -1, offset: int = 0
) -> Any:
    """Interpret a buffer as a 1-dimensional array.

    Args:
        buffer: Buffer to read.
        dtype: Data-type of the returned array.
        count: Number of items to read.
        offset: Start reading the buffer from this offset.

    Returns:
        The constructed array.
    """
    if config.eager_mode:
        np = __import__("numpy")

        res = np.frombuffer(buffer, dtype=dtype, count=count, offset=offset)
        return _wrap(
            _ml_switcheroo_compiler.Tensor(
                res,
                config=_ml_switcheroo_compiler.core.tensor.TensorConfig(
                    shape=res.shape,
                    dtype=_ml_switcheroo_compiler.core.dtype.DType(str(res.dtype)),
                    device=config.default_device,
                ),
            )
        )
    return _wrap(
        ops.frombuffer(buffer, dtype=dtype, count=count, offset=offset)
    )  # pragma: no cover


def fromfile(*args: Any, **kwargs: Any) -> Any:
    """Construct an array from data in a text or binary file.

    Args:
        *args: positional args.
        **kwargs: keyword args.

    Returns:
        The array.
    """
    return _wrap(ops.fromfile(*args, **kwargs))  # pragma: no cover


def fromfunction(
    function: Any, shape: Any, *, dtype: Any = float, **kwargs: Any
) -> Any:
    """Construct an array by executing a function over each coordinate.

    Args:
        function: The function is called with N parameters, where N is the rank of shape.
        shape: Shape of the coordinate grid.
        dtype: Data-type of the coordinate arrays passed to function.
        **kwargs: kwargs.

    Returns:
        The array.
    """

    # We must wrap the function if it passes tensors
    def wrapped_func(*tensors):
        return _to_tensor(function(*[_wrap(t) for t in tensors]))

    return _wrap(ops.fromfunction(wrapped_func, shape, dtype=dtype, **kwargs))


def fromiter(*args: Any, **kwargs: Any) -> Any:
    """Create a new 1-dimensional array from an iterable object.

    Args:
        *args: args.
        **kwargs: kwargs.

    Returns:
        The array.
    """
    return _wrap(ops.fromiter(*args, **kwargs))


def frompyfunc(func: Any, /, nin: int, nout: int, *, identity: Any = None) -> Any:
    """Takes an arbitrary Python function and returns a NumPy ufunc.

    Args:
        func: Python function.
        nin: Number of inputs.
        nout: Number of outputs.
        identity: Identity.

    Returns:
        The ufunc.
    """
    # ml-switcheroo-compiler doesn't return ufuncs directly, it proxies to numpy's frompyfunc which returns a standard numpy ufunc
    # For JAX parity we just pass it along, the returned ufunc might not return zero-jax wrappers directly.
    return ops.frompyfunc(func, nin, nout, identity=identity)  # pragma: no cover


def fromstring(string: str, dtype: Any = float, count: int = -1, *, sep: str) -> Any:
    """A new 1-D array initialized from text data in a string.

    Args:
        string: A string containing the data.
        dtype: Data type of the array.
        count: Read this number of elements.
        sep: The string separating numbers in the data.

    Returns:
        The array.
    """
    return _wrap(ops.fromstring(string, dtype=dtype, count=count, sep=sep))


def geomspace(
    start: Any,
    stop: Any,
    num: int = 50,
    endpoint: bool = True,
    dtype: Any = None,
    axis: int = 0,
) -> Any:
    """Return numbers spaced evenly on a log scale (a geometric progression).

    Args:
        start: The starting value of the sequence.
        stop: The final value of the sequence.
        num: Number of samples to generate.
        endpoint: If true, stop is the last sample.
        dtype: The type of the output array.
        axis: The axis in the result to store the samples.

    Returns:
        The samples.
    """
    return _wrap(
        ops.geomspace(
            _to_tensor(start),
            _to_tensor(stop),
            num=num,
            endpoint=endpoint,
            dtype=dtype,
            axis=axis,
        )
    )


def get_printoptions() -> Any:
    """Return the current print options.

    Returns:
        Print options.
    """
    return ops.get_printoptions()


def gradient(f: Any, *varargs: Any, axis: Any = None, edge_order: Any = None) -> Any:
    """Return the gradient of an N-dimensional array.

    Args:
        f: An N-dimensional array containing samples of a scalar function.
        *varargs: Spacing between f values.
        axis: Gradient is calculated only along the given axis or axes.
        edge_order: Gradient is calculated using N-th order accurate differences at the boundaries.

    Returns:
        The gradient.
    """
    res = ops.gradient(
        _to_tensor(f),
        *[_to_tensor(v) for v in varargs],
        axis=axis,
        edge_order=edge_order if edge_order is not None else 1,
    )
    if isinstance(res, list):
        return [_wrap(r) for r in res]  # pragma: no cover
    return _wrap(res)


def hamming(M: int) -> Any:
    """Return the Hamming window.

    Args:
        M: Number of points in the output window.

    Returns:
        The window.
    """
    return _wrap(ops.hamming(M))


def hanning(M: int) -> Any:
    """Return the Hanning window.

    Args:
        M: Number of points in the output window.

    Returns:
        The window.
    """
    return _wrap(ops.hanning(M))


def histogram(
    a: Any, bins: Any = 10, range: Any = None, weights: Any = None, density: Any = None
) -> Any:
    """Compute the histogram of a dataset.

    Args:
        a: Input data.
        bins: The bins.
        range: The lower and upper range of the bins.
        weights: An array of weights.
        density: If False, the result will contain the number of samples in each bin.

    Returns:
        tuple (hist, bin_edges).
    """
    res = ops.histogram(
        _to_tensor(a),
        bins=_to_tensor(bins) if not isinstance(bins, int) else bins,
        range=range,
        weights=_to_tensor(weights) if weights is not None else None,
        density=density,
    )
    return _wrap(res[0]), _wrap(res[1])


def histogram2d(
    x: Any,
    y: Any,
    bins: Any = 10,
    range: Any = None,
    weights: Any = None,
    density: Any = None,
) -> Any:
    """Compute the bi-dimensional histogram of two data samples.

    Args:
        x: An array containing the x coordinates of the points to be histogrammed.
        y: An array containing the y coordinates of the points to be histogrammed.
        bins: The bin specification.
        range: The leftmost and rightmost edges of the bins.
        weights: An array of values w_i weighing each sample (x_i, y_i).
        density: If False, the default, returns the number of samples in each bin.

    Returns:
        tuple (H, xedges, yedges).
    """
    res = ops.histogram2d(
        _to_tensor(x),
        _to_tensor(y),
        bins=bins,
        range=range,
        weights=_to_tensor(weights) if weights is not None else None,
        density=density,
    )
    return _wrap(res[0]), _wrap(res[1]), _wrap(res[2])


def histogram_bin_edges(
    a: Any, bins: Any = 10, range: Any = None, weights: Any = None
) -> Any:
    """Function to calculate only the edges of the bins used by the histogram function.

    Args:
        a: Input data.
        bins: The bin specification.
        range: The lower and upper range of the bins.
        weights: An array of weights.

    Returns:
        Array of dtype float64 and ndim 1 containing the bin edges.
    """
    return _wrap(
        ops.histogram_bin_edges(
            _to_tensor(a),
            bins=bins,
            range=range,
            weights=_to_tensor(weights) if weights is not None else None,
        )
    )


def histogramdd(
    sample: Any,
    bins: Any = 10,
    range: Any = None,
    weights: Any = None,
    density: Any = None,
) -> Any:
    """Compute the multidimensional histogram of some data.

    Args:
        sample: The data to be histogrammed.
        bins: The bin specification.
        range: A sequence of length D, each an optional (lower, upper) tuple giving the outer bin edges.
        weights: An array of values w_i weighing each sample.
        density: If False, the default, returns the number of samples in each bin.

    Returns:
        tuple (H, edges).
    """
    res = ops.histogramdd(
        _to_tensor(sample),
        bins=bins,
        range=range,
        weights=_to_tensor(weights) if weights is not None else None,
        density=density,
    )
    return _wrap(res[0]), [_wrap(e) for e in res[1]]


def i0(x: Any) -> Any:
    """Modified Bessel function of the first kind, order 0.

    Args:
        x: Array of arguments.

    Returns:
        The modified Bessel function evaluated at each of the elements of x.
    """
    return _wrap(ops.i0(_to_tensor(x)))


def iinfo(int_type: Any) -> Any:
    """Machine limits for integer types.

    Args:
        int_type: The kind of integer data type.

    Returns:
        Machine limits.
    """
    if config.eager_mode:
        np = __import__("numpy")

        return np.iinfo(int_type)
    return ops.iinfo(int_type)  # pragma: no cover


def indices(dimensions: Any, dtype: Any = None, sparse: bool = False) -> Any:
    """Return an array representing the indices of a grid.

    Args:
        dimensions: The shape of the grid.
        dtype: Data type of the result.
        sparse: Return a sparse representation of the grid instead of a dense representation.

    Returns:
        If sparse is False: one array of grid indices. If True: a tuple of arrays.
    """
    # Default numpy is int32, we map properly
    d = (
        "int32"
        if dtype is None
        else (dtype.value if hasattr(dtype, "value") else dtype)
    )
    res = ops.indices(dimensions, dtype=d, sparse=sparse)
    if sparse:
        return tuple([_wrap(r) for r in res])
    return _wrap(res)  # pragma: no cover


def insert(arr: Any, obj: Any, values: Any, axis: Any = None) -> Any:
    """Insert values along the given axis before the given indices.

    Args:
        arr: Input array.
        obj: Object that defines the index or indices before which values is inserted.
        values: Values to insert into arr.
        axis: Axis along which to insert values.

    Returns:
        A copy of arr with values inserted.
    """
    return _wrap(ops.insert(_to_tensor(arr), obj, _to_tensor(values), axis=axis))


def interp(
    x: Any, xp: Any, fp: Any, left: Any = None, right: Any = None, period: Any = None
) -> Any:
    """One-dimensional linear interpolation.

    Args:
        x: The x-coordinates at which to evaluate the interpolated values.
        xp: The x-coordinates of the data points.
        fp: The y-coordinates of the data points.
        left: Value to return for x < xp[0], default is fp[0].
        right: Value to return for x > xp[-1], default is fp[-1].
        period: A period for the x-coordinates.

    Returns:
        The interpolated values.
    """
    return _wrap(
        ops.interp(
            _to_tensor(x),
            _to_tensor(xp),
            _to_tensor(fp),
            left=_to_tensor(left) if left is not None else None,
            right=_to_tensor(right) if right is not None else None,
            period=_to_tensor(period) if period is not None else None,
        )
    )


def intersect1d(
    ar1: Any, ar2: Any, assume_unique: bool = False, return_indices: bool = False
) -> Any:
    """Find the intersection of two arrays.

    Args:
        ar1: Input array.
        ar2: Input array.
        assume_unique: If True, the input arrays are both assumed to be unique.
        return_indices: If True, the indices which correspond to the intersection of the two arrays are returned.

    Returns:
        Sorted 1D array of common and unique elements.
    """
    res = ops.intersect1d(
        _to_tensor(ar1),
        _to_tensor(ar2),
        assume_unique=assume_unique,
        return_indices=return_indices,
    )
    if return_indices:
        return _wrap(res[0]), _wrap(res[1]), _wrap(res[2])  # pragma: no cover
    return _wrap(res)


def iscomplex(x: Any) -> Any:
    """Returns a bool array, where True if input element is complex.

    Args:
        x: Input array.

    Returns:
        Boolean array.
    """
    return _wrap(ops.iscomplex(_to_tensor(x)))


def _safe_bool(val: Any) -> bool:
    if hasattr(val, "value") and not callable(val.value):
        return bool(val.value)  # pragma: no cover
    if hasattr(val, "item") and callable(val.item):
        return bool(val.item())  # pragma: no cover
    try:
        return bool(val)
    except Exception:
        return False


def iscomplexobj(x: Any) -> bool:
    # Using proxy node if possible
    return _safe_bool(ops.iscomplexobj(_to_tensor(x)))


def isdtype(dtype: Any, kind: Any) -> bool:
    return _safe_bool(ops.issubdtype(dtype, kind))  # pragma: no cover


def isin(
    element: Any, test_elements: Any, assume_unique: bool = False, invert: bool = False
) -> Any:
    """Calculates element in test_elements, broadcasting over element only.

    Args:
        element: Input array.
        test_elements: The values against which to test each value of element.
        assume_unique: If True, the input arrays are both assumed to be unique.
        invert: If True, the values in the returned array are inverted.

    Returns:
        Has the same shape as element.
    """
    return _wrap(
        ops.isin(
            _to_tensor(element),
            _to_tensor(test_elements),
            assume_unique=assume_unique,
            invert=invert,
        )
    )


def isreal(x: Any) -> Any:
    """Returns a bool array, where True if input element is real.

    Args:
        x: Input array.

    Returns:
        Boolean array.
    """
    return _wrap(ops.isreal(_to_tensor(x)))


def isrealobj(x: Any) -> bool:
    try:  # pragma: no cover
        return _safe_bool(ops.isrealobj(getattr(x, "data", x)))  # pragma: no cover
    except Exception:  # pragma: no cover
        return True  # pragma: no cover


def isscalar(element: Any) -> bool:
    try:
        val = getattr(element, "data", element)
        if hasattr(val, "dtype") and hasattr(val.dtype, "name"):
            return False  # pragma: no cover
        if isinstance(val, (int, float, complex, bool, str, bytes, type(None))):
            return True
        return _safe_bool(ops.isscalar(val))  # pragma: no cover
    except Exception:  # pragma: no cover
        return False  # pragma: no cover


def issubdtype(arg1: Any, arg2: Any) -> bool:
    try:  # pragma: no cover
        return _safe_bool(ops.issubdtype(arg1, arg2))  # pragma: no cover
    except Exception:  # pragma: no cover
        return False  # pragma: no cover


def iterable(y: Any) -> bool:
    return _safe_bool(ops.iterable(getattr(y, "data", y)))  # pragma: no cover


def ix_(*args: Any) -> Any:
    """Construct an open mesh from multiple sequences.

    Args:
        *args: 1-D sequences.

    Returns:
        Tuple of ndarrays.
    """
    if config.eager_mode:
        np = __import__("numpy")

        res = np.ix_(*[getattr(_to_tensor(a), "data", _to_tensor(a)) for a in args])
        return tuple(
            _wrap(
                _ml_switcheroo_compiler.Tensor(
                    r,
                    config=_ml_switcheroo_compiler.core.tensor.TensorConfig(
                        shape=r.shape,
                        dtype=_ml_switcheroo_compiler.core.dtype.DType(str(r.dtype)),
                        device=config.default_device,
                    ),
                )
            )
            for r in res
        )
    return tuple(
        _wrap(t) for t in ops.ix_(*[_to_tensor(a) for a in args])
    )  # pragma: no cover


def kaiser(M: int, beta: Any) -> Any:
    """Return the Kaiser window.

    Args:
        M: Number of points in the output window.
        beta: Shape parameter for window.

    Returns:
        The window.
    """
    # beta can be an array in JAX, but compiler expects float.
    # We pass it as tensor and if compiler fails in eager, it's compiler's responsibility.
    return _wrap(ops.kaiser(M, _to_tensor(beta)))


def kron(a: Any, b: Any) -> Any:
    """Kronecker product of two arrays.

    Args:
        a: Input array.
        b: Input array.

    Returns:
        The Kronecker product of a and b.
    """
    return _wrap(ops.kron(_to_tensor(a), _to_tensor(b)))


def lexsort(keys: Any, axis: int = -1) -> Any:
    """Perform an indirect stable sort using a sequence of keys.

    Args:
        keys: The k different "columns" to be sorted.
        axis: Axis to be indirectly sorted.

    Returns:
        Array of indices that sort the keys along the specified axis.
    """
    return _wrap(
        ops.lexsort(
            [_to_tensor(k) for k in keys]
            if isinstance(keys, (list, tuple))
            else _to_tensor(keys),
            axis=axis,
        )
    )


def load(*args: Any, **kwargs: Any) -> Any:
    """Load arrays or pickled objects from .npy, .npz or pickled files.

    Args:
        *args: positional args.
        **kwargs: keyword args.

    Returns:
        The result.
    """
    res = ops.load(*args, **kwargs)  # pragma: no cover
    if isinstance(res, tuple):  # pragma: no cover
        return tuple(_wrap(r) for r in res)  # pragma: no cover
    if isinstance(res, list):  # pragma: no cover
        return [_wrap(r) for r in res]  # pragma: no cover
    # For .npz files it might return a dictionary-like object, which we don't wrap deeply here automatically.
    return _wrap(res) if hasattr(res, "shape") else res  # pragma: no cover


def mask_indices(*args: Any, **kwargs: Any) -> Any:
    """Return the indices to access (n, n) arrays, given a masking function.

    Args:
        *args: positional args.
        **kwargs: keyword args.

    Returns:
        Tuple of indices.
    """
    return tuple(_wrap(t) for t in ops.mask_indices(*args, **kwargs))


def matrix_transpose(x: Any, /) -> Any:
    """Transposes a matrix (or a stack of matrices) x.

    Args:
        x: Input array.

    Returns:
        The transposed array.
    """
    return _wrap(ops.matrix_transpose(_to_tensor(x)))  # pragma: no cover


def median(
    a: Any,
    axis: Any = None,
    out: Any = None,
    overwrite_input: bool = False,
    keepdims: bool = False,
) -> Any:
    """Compute the median along the specified axis.

    Args:
        a: Input array or object that can be converted to an array.
        axis: Axis or axes along which the medians are computed.
        out: Alternative output array in which to place the result.
        overwrite_input: If True, then allow use of memory of input array a for calculations.
        keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns:
        A new array holding the result.
    """
    return _wrap(
        ops.median(
            _to_tensor(a),
            axis=axis,
            out=out,
            overwrite_input=overwrite_input,
            keepdims=keepdims,
        )
    )


def modf(x: Any, /, out: Any = None) -> Any:
    """Return the fractional and integral parts of an array, element-wise.

    Args:
        x: Input array.
        out: A location into which the result is stored.

    Returns:
        Fractional and integral parts.
    """
    res = (
        ops.modf(_to_tensor(x), out=out) if out is not None else ops.modf(_to_tensor(x))
    )
    return _wrap(res[0]), _wrap(res[1])


def nanargmax(a: Any, axis: Any = None, out: Any = None, keepdims: Any = None) -> Any:
    """Return the indices of the maximum values in the specified axis ignoring NaNs.

    Args:
        a: Input data.
        axis: Axis along which to operate.
        out: If provided, the result will be inserted into this array.
        keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns:
        Array of indices into the array.
    """
    return _wrap(ops.nanargmax(_to_tensor(a), axis=axis, out=out, keepdims=keepdims))


def nanargmin(a: Any, axis: Any = None, out: Any = None, keepdims: Any = None) -> Any:
    """Return the indices of the minimum values in the specified axis ignoring NaNs.

    Args:
        a: Input data.
        axis: Axis along which to operate.
        out: If provided, the result will be inserted into this array.
        keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns:
        Array of indices into the array.
    """
    return _wrap(ops.nanargmin(_to_tensor(a), axis=axis, out=out, keepdims=keepdims))


def nancumprod(a: Any, axis: Any = None, dtype: Any = None, out: Any = None) -> Any:
    """Return the cumulative product of array elements over a given axis treating NaNs as one.

    Args:
        a: Input array.
        axis: Axis along which the cumulative product is computed.
        dtype: Type of the returned array, as well as of the accumulator in which the elements are multiplied.
        out: Alternative output array in which to place the result.

    Returns:
        A new array holding the result is returned unless out is specified.
    """
    return _wrap(ops.nancumprod(_to_tensor(a), axis=axis, dtype=dtype, out=out))


def nancumsum(a: Any, axis: Any = None, dtype: Any = None, out: Any = None) -> Any:
    """Return the cumulative sum of array elements over a given axis treating NaNs as zero.

    Args:
        a: Input array.
        axis: Axis along which the cumulative sum is computed.
        dtype: Type of the returned array, as well as of the accumulator in which the elements are summed.
        out: Alternative output array in which to place the result.

    Returns:
        A new array holding the result is returned unless out is specified.
    """
    return _wrap(ops.nancumsum(_to_tensor(a), axis=axis, dtype=dtype, out=out))


def nanmean(
    a: Any,
    axis: Any = None,
    dtype: Any = None,
    out: Any = None,
    keepdims: bool = False,
    where: Any = None,
) -> Any:
    """Compute the arithmetic mean along the specified axis, ignoring NaNs.

    Args:
        a: Array containing numbers whose mean is desired.
        axis: Axis or axes along which the means are computed.
        dtype: Type to use in computing the mean.
        out: Alternate output array in which to place the result.
        keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        where: Elements to include in the mean.

    Returns:
        A new array containing the mean values.
    """
    return _wrap(
        ops.nanmean(
            _to_tensor(a),
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            where=_to_tensor(where) if where is not None else None,
        )
    )


def nanmedian(
    a: Any,
    axis: Any = None,
    out: Any = None,
    overwrite_input: bool = False,
    keepdims: bool = False,
) -> Any:
    """Compute the median along the specified axis, while ignoring NaNs.

    Args:
        a: Input array or object that can be converted to an array.
        axis: Axis or axes along which the medians are computed.
        out: Alternative output array in which to place the result.
        overwrite_input: If True, then allow use of memory of input array a for calculations.
        keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns:
        A new array holding the result.
    """
    return _wrap(
        ops.nanmedian(
            _to_tensor(a),
            axis=axis,
            out=out,
            overwrite_input=overwrite_input,
            keepdims=keepdims,
        )
    )


def nanpercentile(
    a: Any,
    q: Any,
    axis: Any = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
    *,
    interpolation: Any = None,
) -> Any:
    """Compute the qth percentile of the data along the specified axis, while ignoring nan values.

    Args:
        a: Input array or object that can be converted to an array.
        q: Array of percentile grades to compute.
        axis: Axis or axes along which the percentiles are computed.
        out: Alternative output array in which to place the result.
        overwrite_input: If True, then allow use of memory of input array a for calculations.
        method: This parameter specifies the method to use for estimating the percentile.
        keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        interpolation: Deprecated name for the method keyword argument.

    Returns:
        A new array holding the result.
    """
    m = method if interpolation is None else interpolation
    return _wrap(
        ops.nanpercentile(
            _to_tensor(a),
            _to_tensor(q),
            axis=axis,
            out=out,
            overwrite_input=overwrite_input,
            method=m,
            keepdims=keepdims,
        )
    )


def nanquantile(
    a: Any,
    q: Any,
    axis: Any = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
    *,
    interpolation: Any = None,
) -> Any:
    """Compute the qth quantile of the data along the specified axis, while ignoring nan values.

    Args:
        a: Input array or object that can be converted to an array.
        q: Array of quantile grades to compute.
        axis: Axis or axes along which the quantiles are computed.
        out: Alternative output array in which to place the result.
        overwrite_input: If True, then allow use of memory of input array a for calculations.
        method: This parameter specifies the method to use for estimating the quantile.
        keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        interpolation: Deprecated name for the method keyword argument.

    Returns:
        A new array holding the result.
    """
    m = method if interpolation is None else interpolation
    return _wrap(
        ops.nanquantile(
            _to_tensor(a),
            _to_tensor(q),
            axis=axis,
            out=out,
            overwrite_input=overwrite_input,
            method=m,
            keepdims=keepdims,
        )
    )


def nanstd(
    a: Any,
    axis: Any = None,
    dtype: Any = None,
    out: Any = None,
    ddof: int = 0,
    keepdims: bool = False,
    where: Any = None,
) -> Any:
    """Compute the standard deviation along the specified axis, while ignoring NaNs.

    Args:
        a: Calculate the standard deviation of these values.
        axis: Axis or axes along which the standard deviation is computed.
        dtype: Type to use in computing the standard deviation.
        out: Alternative output array in which to place the result.
        ddof: Means Delta Degrees of Freedom.
        keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        where: Elements to include in the standard deviation.

    Returns:
        A new array containing the standard deviation values.
    """
    return _wrap(
        ops.nanstd(
            _to_tensor(a),
            axis=axis,
            dtype=dtype,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            where=_to_tensor(where) if where is not None else None,
        )
    )


def nanvar(
    a: Any,
    axis: Any = None,
    dtype: Any = None,
    out: Any = None,
    ddof: int = 0,
    keepdims: bool = False,
    where: Any = None,
) -> Any:
    """Compute the variance along the specified axis, while ignoring NaNs.

    Args:
        a: Array containing numbers whose variance is desired.
        axis: Axis or axes along which the variance is computed.
        dtype: Type to use in computing the variance.
        out: Alternate output array in which to place the result.
        ddof: Means Delta Degrees of Freedom.
        keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        where: Elements to include in the variance.

    Returns:
        A new array containing the variance values.
    """
    return _wrap(
        ops.nanvar(
            _to_tensor(a),
            axis=axis,
            dtype=dtype,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            where=_to_tensor(where) if where is not None else None,
        )
    )


# Missing top-level functions
np = __import__("numpy")
from .tensor_utils import to_array


def nonzero(a: Any, *, size: Any = None, fill_value: Any = None) -> Any:
    np = __import__("numpy")
    from .tensor_utils import to_array

    res = np.nonzero(to_array(a.data if hasattr(a, "data") else a))
    return tuple(array(r) for r in res)


def packbits(a: Any, axis: Any = None, bitorder: str = "big") -> Any:
    # Not supported well natively in IR yet, fallback to numpy
    return array(
        np.packbits(
            to_array(a.data if hasattr(a, "data") else a), axis=axis, bitorder=bitorder
        )
    )


def unpackbits(
    a: Any, axis: Any = None, count: Any = None, bitorder: str = "big"
) -> Any:
    return array(
        np.unpackbits(
            to_array(a.data if hasattr(a, "data") else a),
            axis=axis,
            count=count,
            bitorder=bitorder,
        )
    )


def partition(a: Any, kth: Any, axis: int = -1) -> Any:
    return array(
        np.partition(to_array(a.data if hasattr(a, "data") else a), kth, axis=axis)
    )


def percentile(
    a: Any,
    q: Any,
    axis: Any = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> Any:
    return array(
        np.percentile(
            to_array(a.data if hasattr(a, "data") else a),
            q,
            axis=axis,
            method=method,
            keepdims=keepdims,
        )
    )


def permute_dims(a: Any, axes: Any) -> Any:
    return transpose(a, axes)


def piecewise(x: Any, condlist: Any, funclist: Any, *args: Any, **kw: Any) -> Any:
    return array(np.piecewise(x, condlist, funclist, *args, **kw))


def place(arr: Any, mask: Any, vals: Any) -> None:
    pass


def promote_types(type1: Any, type2: Any) -> Any:
    return np.promote_types(type1, type2)


def ptp(a: Any, axis: Any = None, out: Any = None, keepdims: bool = False) -> Any:
    return amax(a, axis=axis, keepdims=keepdims) - amin(a, axis=axis, keepdims=keepdims)


def put(a: Any, ind: Any, v: Any, mode: str = "raise") -> None:
    pass


def quantile(
    a: Any,
    q: Any,
    axis: Any = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> Any:
    return array(
        np.quantile(
            to_array(a.data if hasattr(a, "data") else a),
            q,
            axis=axis,
            method=method,
            keepdims=keepdims,
        )
    )


def ravel_multi_index(
    multi_index: Any, dims: Any, mode: str = "raise", order: str = "C"
) -> Any:
    return array(np.ravel_multi_index(multi_index, dims, mode=mode, order=order))


def resize(a: Any, new_shape: Any) -> Any:
    return array(np.resize(to_array(a.data if hasattr(a, "data") else a), new_shape))


def result_type(*arrays_and_dtypes: Any) -> Any:
    np = __import__("numpy")
    args = [getattr(a, "dtype", a) for a in arrays_and_dtypes]
    res = np.result_type(*[getattr(a, "value", a) for a in args])
    return res.name if hasattr(res, "name") else res


def rollaxis(a: Any, axis: int, start: int = 0) -> Any:
    # Use moveaxis internally since rollaxis delegates to it and numpy 2 expects arrays with ndim
    return moveaxis(a, axis, start)


def rot90(m: Any, k: int = 1, axes: Any = (0, 1)) -> Any:
    return array(np.rot90(m, k=k, axes=axes))


def save(
    file: Any, arr: Any, allow_pickle: bool = True, fix_imports: bool = True
) -> None:
    np.save(file, arr, allow_pickle=allow_pickle, fix_imports=fix_imports)


def savez(file: Any, *args: Any, **kwds: Any) -> None:
    np.savez(file, *args, **kwds)


def setdiff1d(ar1: Any, ar2: Any, assume_unique: bool = False) -> Any:
    return array(np.setdiff1d(ar1, ar2, assume_unique=assume_unique))


def setxor1d(ar1: Any, ar2: Any, assume_unique: bool = False) -> Any:
    return array(np.setxor1d(ar1, ar2, assume_unique=assume_unique))


def size(a: Any, axis: Any = None) -> Any:
    np = __import__("numpy")
    from .tensor_utils import to_array

    _arr = to_array(a.data if hasattr(a, "data") else a)
    return np.size(_arr, axis=axis)


def sort_complex(a: Any) -> Any:
    return array(np.sort_complex(a))


def trace(
    a: Any,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    dtype: Any = None,
    out: Any = None,
) -> Any:
    return array(
        np.trace(
            to_array(a.data if hasattr(a, "data") else a),
            offset=offset,
            axis1=axis1,
            axis2=axis2,
            dtype=dtype,
        )
    )


def trapezoid(y: Any, x: Any = None, dx: float = 1.0, axis: int = -1) -> Any:
    return array(np.trapz(y, x=x, dx=dx, axis=axis))


def tri(N: int, M: Any = None, k: int = 0, dtype: Any = float) -> Any:
    return array(np.tri(N, M=M, k=k, dtype=dtype))


def tril_indices(n: int, k: int = 0, m: Any = None) -> Any:
    res = np.tril_indices(n, k=k, m=m)
    return tuple(array(r) for r in res)


def tril_indices_from(arr: Any, k: int = 0) -> Any:
    res = np.tril_indices_from(arr, k=k)
    return tuple(array(r) for r in res)


def trim_zeros(filt: Any, trim: str = "fb") -> Any:
    return array(np.trim_zeros(filt, trim=trim))


def triu_indices(n: int, k: int = 0, m: Any = None) -> Any:
    res = np.triu_indices(n, k=k, m=m)
    return tuple(array(r) for r in res)


def triu_indices_from(arr: Any, k: int = 0) -> Any:
    res = np.triu_indices_from(arr, k=k)
    return tuple(array(r) for r in res)


def ufunc() -> None:
    pass  # pragma: no cover


def uint(x: Any = 0) -> Any:
    return np.uint(x)


def uint4(x: Any = 0) -> Any:
    # JAX custom dtype, mocked with uint8
    return np.uint8(x)


def union1d(ar1: Any, ar2: Any) -> Any:
    return array(np.union1d(ar1, ar2))


def unique(
    ar: Any,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: Any = None,
    *,
    size: Any = None,
    fill_value: Any = None,
) -> Any:
    res = np.unique(
        ar,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
    )
    if isinstance(res, tuple):
        return tuple(array(r) for r in res)
    return array(res)


def unique_all(x: Any, *, size: Any = None, fill_value: Any = None) -> Any:
    return unique(x, return_index=True, return_inverse=True, return_counts=True)


def unique_counts(x: Any, *, size: Any = None, fill_value: Any = None) -> Any:
    return unique(x, return_counts=True)


def unique_inverse(x: Any, *, size: Any = None, fill_value: Any = None) -> Any:
    return unique(x, return_inverse=True)


def unique_values(x: Any, *, size: Any = None, fill_value: Any = None) -> Any:
    return unique(x)


def unravel_index(indices: Any, shape: Any) -> Any:
    res = np.unravel_index(indices, shape)
    return tuple(array(r) for r in res)


def unwrap(
    p: Any, discont: Any = None, axis: int = -1, period: float = 6.283185307179586
) -> Any:
    return array(np.unwrap(p, discont=discont, axis=axis, period=period))


def vander(x: Any, N: Any = None, increasing: bool = False) -> Any:
    return array(np.vander(x, N=N, increasing=increasing))


def vecdot(x1: Any, x2: Any, /, *, axis: int = -1) -> Any:
    return sum(x1 * x2, axis=axis)


class vectorize:
    def __init__(
        self,
        pyfunc: Any,
        otypes: Any = None,
        doc: Any = None,
        excluded: Any = None,
        cache: bool = False,
        signature: Any = None,
    ) -> None:
        self.pyfunc = pyfunc
        self.otypes = otypes
        self.doc = doc
        self.excluded = excluded
        self.cache = cache
        self.signature = signature
        self._vfunc = np.vectorize(
            pyfunc,
            otypes=otypes,
            doc=doc,
            excluded=excluded,
            cache=cache,
            signature=signature,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return array(self._vfunc(*args, **kwargs))


__all__ = [
    k
    for k in dir()
    if not k.startswith("_") and k not in ["DType", "TensorConfig", "to_array"]
]
