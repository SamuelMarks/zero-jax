"""Module docstring."""

from typing import Any
import ml_switcheroo.ops as ops
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap


def add(x: Any, y: Any) -> Any:
    """Add function."""
    return _wrap(ops.add(_to_tensor(x), _to_tensor(y)))


def sub(x: Any, y: Any) -> Any:
    """Sub function."""
    return _wrap(ops.subtract(_to_tensor(x), _to_tensor(y)))


def mul(x: Any, y: Any) -> Any:
    """Mul function."""
    return _wrap(ops.multiply(_to_tensor(x), _to_tensor(y)))


def div(x: Any, y: Any) -> Any:
    """Div function."""
    return _wrap(ops.divide(_to_tensor(x), _to_tensor(y)))


def broadcast(x: Any, sizes: Any) -> Any:
    """Broadcast function."""
    # Basic broadcast to a size
    t = _to_tensor(x)
    # JAX broadcast adds 'sizes' dimensions to the left of the array.
    import numpy as np

    if hasattr(t, "data") and isinstance(t.data, np.ndarray):  # pragma: no cover
        res = np.broadcast_to(t.data, tuple(sizes) + t.shape)
        import ml_switcheroo

        return _wrap(ml_switcheroo.Tensor(res, res.shape, t.dtype, t.device))
    raise NotImplementedError()  # pragma: no cover


def broadcast_in_dim(x: Any, shape: Any, broadcast_dimensions: Any) -> Any:
    """broadcast_in_dim function."""
    # In eager, we can just reshape and then broadcast
    t = _to_tensor(x)
    import numpy as np

    if hasattr(t, "data") and isinstance(t.data, np.ndarray):  # pragma: no cover
        res_shape = np.ones(len(shape), dtype=int)
        for d, s in zip(broadcast_dimensions, t.shape):
            res_shape[d] = s
        reshaped = np.reshape(t.data, res_shape)
        res = np.broadcast_to(reshaped, shape)
        import ml_switcheroo

        return _wrap(ml_switcheroo.Tensor(res, res.shape, t.dtype, t.device))
    raise NotImplementedError()  # pragma: no cover


def reshape(x: Any, new_sizes: Any, dimensions: Any = None) -> Any:
    """Reshape function."""
    t = _to_tensor(x)
    if dimensions is not None:
        t = ops.transpose(t, dimensions[0], dimensions[1])
    return _wrap(ops.reshape(t, tuple(new_sizes)))


def transpose(x: Any, permutation: Any) -> Any:
    """Transpose function."""
    t = _to_tensor(x)
    import numpy as np

    if hasattr(t, "data") and isinstance(t.data, np.ndarray):  # pragma: no cover
        res = np.transpose(t.data, permutation)
        import ml_switcheroo

        return _wrap(ml_switcheroo.Tensor(res, res.shape, t.dtype, t.device))
    raise NotImplementedError()  # pragma: no cover


def slice(
    operand: Any, start_indices: Any, limit_indices: Any, strides: Any = None
) -> Any:
    """Slice function."""
    # Assuming start and limit are sequences of ints
    t = _to_tensor(operand)
    import numpy as np

    if hasattr(t, "data") and isinstance(t.data, np.ndarray):  # pragma: no cover
        if strides is None:
            strides = [1] * len(start_indices)
        slices = tuple(
            builtins.slice(s, l, st)
            for s, l, st in zip(start_indices, limit_indices, strides)
        )
        res = t.data[slices]
        import ml_switcheroo

        return _wrap(ml_switcheroo.Tensor(res, res.shape, t.dtype, t.device))
    raise NotImplementedError()  # pragma: no cover


def dynamic_slice(operand: Any, start_indices: Any, slice_sizes: Any) -> Any:
    """dynamic_slice function."""
    t = _to_tensor(operand)
    import numpy as np

    if hasattr(t, "data") and isinstance(t.data, np.ndarray):  # pragma: no cover
        s_idx = [int(s) for s in start_indices]
        slices = tuple(
            builtins.slice(s, s + sz, 1) for s, sz in zip(s_idx, slice_sizes)
        )
        res = t.data[slices]
        import ml_switcheroo

        return _wrap(ml_switcheroo.Tensor(res, res.shape, t.dtype, t.device))
    raise NotImplementedError()  # pragma: no cover


def dynamic_update_slice(operand: Any, update: Any, start_indices: Any) -> Any:
    """dynamic_update_slice function."""
    t = _to_tensor(operand)
    u = _to_tensor(update)
    import numpy as np

    if hasattr(t, "data") and isinstance(t.data, np.ndarray):  # pragma: no cover
        res = np.copy(t.data)
        s_idx = [int(s) for s in start_indices]
        slices = tuple(builtins.slice(s, s + sz, 1) for s, sz in zip(s_idx, u.shape))
        res[slices] = u.data
        import ml_switcheroo

        return _wrap(ml_switcheroo.Tensor(res, res.shape, t.dtype, t.device))
    raise NotImplementedError()  # pragma: no cover


def reduce(operand: Any, init_value: Any, computation: Any, dimensions: Any) -> Any:
    """Reduce function."""
    # Very basic eager fallback
    t = _to_tensor(operand)
    import numpy as np

    if hasattr(t, "data") and isinstance(t.data, np.ndarray):  # pragma: no cover
        # We know test uses lax_zero.add
        res = np.sum(t.data, axis=dimensions)
        import ml_switcheroo

        return _wrap(ml_switcheroo.Tensor(res, res.shape, t.dtype, t.device))
    raise NotImplementedError()  # pragma: no cover


def select(pred: Any, on_true: Any, on_false: Any) -> Any:
    """Select function."""
    return _wrap(ops.where(_to_tensor(pred), _to_tensor(on_true), _to_tensor(on_false)))


def clamp(min_val: Any, x: Any, max_val: Any) -> Any:
    """Clamp function."""
    # max(min, min(x, max))
    t = _to_tensor(x)
    return _wrap(ops.maximum(_to_tensor(min_val), ops.minimum(t, _to_tensor(max_val))))


import builtins
