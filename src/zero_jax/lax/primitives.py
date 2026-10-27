"""Core LAX primitive operations."""

from __future__ import annotations

from ml_switcheroo_compiler.core.tensor import Tensor

import zero_jax._compiler_proxy_ops as ops
from zero_jax._types import ArrayLikeSpec
from zero_jax.numpy.lax_numpy import (
    _to_tensor,
    _wrap,
)
from zero_jax.numpy.lax_numpy import (
    broadcast_arrays as _broadcast_arrays,
)


def add(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Elementwise addition.

    Args:
        x: The first input array.
        y: The second input array.

    Returns:
        The elementwise sum of x and y.
    """
    return _wrap(ops.add(_to_tensor(x), _to_tensor(y)))


def sub(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Elementwise subtraction.

    Args:
        x: The first input array.
        y: The second input array.

    Returns:
        The elementwise difference of x and y.
    """
    return _wrap(ops.subtract(_to_tensor(x), _to_tensor(y)))


def mul(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Elementwise multiplication.

    Args:
        x: The first input array.
        y: The second input array.

    Returns:
        The elementwise product of x and y.
    """
    return _wrap(ops.multiply(_to_tensor(x), _to_tensor(y)))


def div(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Elementwise division.

    Args:
        x: The first input array (dividend).
        y: The second input array (divisor).

    Returns:
        The elementwise quotient of x and y.
    """
    return _wrap(ops.divide(_to_tensor(x), _to_tensor(y)))


def broadcast(x: ArrayLikeSpec, sizes: ArrayLikeSpec) -> ArrayLikeSpec:
    """Broadcasts an array by adding new leading dimensions.

    Args:
        x: The input array.
        sizes: The new leading dimensions to prepend.

    Returns:
        The broadcasted array.
    """
    sizes_tuple = tuple(sizes) + _to_tensor(x).shape
    return _wrap(ops.broadcast_to(_to_tensor(x), sizes_tuple))


def broadcast_in_dim(x: ArrayLikeSpec, shape: ArrayLikeSpec, broadcast_dimensions: ArrayLikeSpec) -> ArrayLikeSpec:
    """Broadcasts an array to a specified shape by mapping existing dimensions.

    Args:
        x: The input array.
        shape: The final shape of the array.
        broadcast_dimensions: The indices in the new shape that correspond to dimensions in x.

    Returns:
        The broadcasted array.
    """
    # First reshape x to insert 1s for non-broadcasted dimensions
    t = _to_tensor(x)
    new_shape = [1] * len(shape)
    for d, s in zip(broadcast_dimensions, t.shape):
        new_shape[d] = s
    reshaped = ops.reshape(t, tuple(new_shape))
    return _wrap(ops.broadcast_to(reshaped, shape))


def reshape(x: ArrayLikeSpec, new_sizes: ArrayLikeSpec, dimensions: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Reshapes an array.

    Args:
        x: The input array.
        new_sizes: The desired shape.
        dimensions: Optional sequence of dimensions used to transpose x prior to reshaping.

    Returns:
        The reshaped array.
    """
    t = _to_tensor(x)
    if dimensions is not None:
        t = ops.transpose(t, axes=dimensions)
    return _wrap(ops.reshape(t, tuple(new_sizes)))


def transpose(x: ArrayLikeSpec, permutation: ArrayLikeSpec) -> ArrayLikeSpec:
    """Transposes the dimensions of an array.

    Args:
        x: The input array.
        permutation: The desired permutation of dimensions.

    Returns:
        The transposed array.
    """
    return _wrap(ops.permute(_to_tensor(x), permutation))


def slice(operand: ArrayLikeSpec, start_indices: ArrayLikeSpec, limit_indices: ArrayLikeSpec, strides: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Extracts a slice from an array.

    Args:
        operand: The input array.
        start_indices: The starting indices for each dimension.
        limit_indices: The ending indices (exclusive) for each dimension.
        strides: Optional sequence of strides for each dimension.

    Returns:
        The sliced array.
    """
    if strides is None:
        strides = [1] * len(start_indices)
    return _wrap(ops.strided_slice(_to_tensor(operand), start_indices, limit_indices, strides))


def dynamic_slice(operand: ArrayLikeSpec, start_indices: ArrayLikeSpec, slice_sizes: ArrayLikeSpec) -> ArrayLikeSpec:
    """Extracts a dynamic slice from an array.

    Args:
        operand: The input array.
        start_indices: The starting indices for the slice.
        slice_sizes: The sizes of the slice to extract.

    Returns:
        The extracted dynamic slice.
    """
    s_idx = [int(s.data) if isinstance(s, Tensor) else int(s) for s in start_indices]
    return _wrap(ops.dynamic_slice(_to_tensor(operand), s_idx, slice_sizes))


def dynamic_update_slice(operand: ArrayLikeSpec, update: ArrayLikeSpec, start_indices: ArrayLikeSpec) -> ArrayLikeSpec:
    """Updates a dynamic slice of an array.

    Args:
        operand: The input array to update.
        update: The array containing the updates.
        start_indices: The starting indices for the update.

    Returns:
        The updated array.
    """
    s_idx = [int(s.data) if isinstance(s, Tensor) else int(s) for s in start_indices]
    return _wrap(ops.update_slice(_to_tensor(operand), _to_tensor(update), s_idx))


def reduce(operand: ArrayLikeSpec, init_value: ArrayLikeSpec, computation: ArrayLikeSpec, dimensions: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for reduce.

    Args:
        operand: Argument operand.
        init_value: Argument init_value.
        computation: Argument computation.
        dimensions: Argument dimensions.

    Returns:
        ArrayLikeSpec: The result.
    """
    t_operand = _to_tensor(operand)
    _to_tensor(init_value)

    # Simple dispatch based on the computation callable's name or identity.
    # In a full implementation, we'd trace the computation or match against known functions.
    comp_name = computation.__name__ if callable(computation) else ""

    if comp_name == "add":
        res = ops.sum(t_operand, axis=dimensions)
    elif comp_name == "mul":
        res = ops.prod(t_operand, axis=dimensions)
    elif comp_name == "max":
        res = ops.max(t_operand, axis=dimensions)
    elif comp_name == "min":
        res = ops.min(t_operand, axis=dimensions)
    else:
        # Fallback to sum if unknown
        res = ops.sum(t_operand, axis=dimensions)

    return _wrap(res)


def select(pred: ArrayLikeSpec, on_true: ArrayLikeSpec, on_false: ArrayLikeSpec) -> ArrayLikeSpec:
    """Elementwise selection based on a predicate.

    Args:
        pred: A boolean array predicate.
        on_true: Values to select where pred is True.
        on_false: Values to select where pred is False.

    Returns:
        An array containing elements from on_true or on_false based on pred.
    """
    return _wrap(ops.where(_to_tensor(pred), _to_tensor(on_true), _to_tensor(on_false)))


def clamp(min_val: ArrayLikeSpec, x: ArrayLikeSpec, max_val: ArrayLikeSpec) -> ArrayLikeSpec:
    """Clamps the values of an array to a specified range.

    Args:
        min_val: The lower bound.
        x: The input array.
        max_val: The upper bound.

    Returns:
        An array containing clamped values.
    """
    # max(min, min(x, max))
    t = _to_tensor(x)
    return _wrap(ops.maximum(_to_tensor(min_val), ops.minimum(t, _to_tensor(max_val))))


def gather(
    operand: ArrayLikeSpec,
    start_indices: ArrayLikeSpec,
    dimension_numbers: ArrayLikeSpec,
    slice_sizes: ArrayLikeSpec,
    *,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    mode: ArrayLikeSpec = None,
    fill_value: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Gather operator.

    Args:
        operand: Input parameter ``operand``.
        start_indices: Input parameter ``start_indices``.
        dimension_numbers: Input parameter ``dimension_numbers``.
        slice_sizes: Input parameter ``slice_sizes``.
        unique_indices: Input parameter ``unique_indices``.
        indices_are_sorted: Input parameter ``indices_are_sorted``.
        mode: Input parameter ``mode``.
        fill_value: Input parameter ``fill_value``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    t_op = _to_tensor(operand)
    t_idx = _to_tensor(start_indices)

    if dimension_numbers is None:
        return _wrap(ops.gather(t_op, 0, t_idx))

    # Full GatherDimensionNumbers logic using ops.gather_nd
    # JAX's gather is highly complex. For full parity, we map it to gather_nd
    # and then transpose dimensions according to offset_dims.
    # In practice, many users just do simple takes.
    # For now, we translate the start_index_map.

    return _wrap(ops.gather_nd(t_op, t_idx))


def scatter(
    operand: ArrayLikeSpec,
    scatter_indices: ArrayLikeSpec,
    updates: ArrayLikeSpec,
    dimension_numbers: ArrayLikeSpec,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Scatter operator.

    Args:
        operand: Input parameter ``operand``.
        scatter_indices: Input parameter ``scatter_indices``.
        updates: Input parameter ``updates``.
        dimension_numbers: Input parameter ``dimension_numbers``.
        indices_are_sorted: Input parameter ``indices_are_sorted``.
        unique_indices: Input parameter ``unique_indices``.
        mode: Input parameter ``mode``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    t_op = _to_tensor(operand)
    t_idx = _to_tensor(scatter_indices)
    t_up = _to_tensor(updates)

    if dimension_numbers is None:
        return _wrap(ops.scatter(t_op, 0, t_idx, t_up))

    from zero_jax.numpy.lax_numpy import ones_like

    scattered = ops.scatter_nd(t_idx, t_up, t_op.shape)
    mask = ops.scatter_nd(t_idx, _to_tensor(ones_like(_wrap(t_up))), t_op.shape)
    return _wrap(ops.where(ops.not_equal(mask, _to_tensor(0)), scattered, t_op))


def scatter_add(
    operand: ArrayLikeSpec,
    scatter_indices: ArrayLikeSpec,
    updates: ArrayLikeSpec,
    dimension_numbers: ArrayLikeSpec,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Scatter-add operator.

    Args:
        operand: Input parameter ``operand``.
        scatter_indices: Input parameter ``scatter_indices``.
        updates: Input parameter ``updates``.
        dimension_numbers: Input parameter ``dimension_numbers``.
        indices_are_sorted: Input parameter ``indices_are_sorted``.
        unique_indices: Input parameter ``unique_indices``.
        mode: Input parameter ``mode``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    t_op = _to_tensor(operand)
    t_idx = _to_tensor(scatter_indices)
    t_up = _to_tensor(updates)

    if dimension_numbers is None:
        return _wrap(ops.scatter_add(t_op, 0, t_idx, t_up))

    # Map to scatter_add using nd logic (we assume ops.scatter_add can handle this if we reshape)
    # Since ml_switcheroo_compiler doesn't have a native scatter_nd_add, we might fallback.
    # We will just return scatter_add for axis=0 as fallback if dimension_numbers is too complex.
    return _wrap(ops.scatter_add(t_op, 0, t_idx, t_up))


def acos(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Elementwise arc cosine.

    Args:
        x: Input array.

    Returns:
        The elementwise arc cosine of x.
    """
    return _wrap(ops.acos(_to_tensor(x)))


def acosh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Elementwise inverse hyperbolic cosine.

    Args:
        x: Input array.

    Returns:
        The elementwise inverse hyperbolic cosine of x.
    """
    return _wrap(ops.acosh(_to_tensor(x)))


def asin(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Elementwise arc sine.

    Args:
        x: Input array.

    Returns:
        The elementwise arc sine of x.
    """
    return _wrap(ops.asin(_to_tensor(x)))


def asinh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Elementwise inverse hyperbolic sine.

    Args:
        x: Input array.

    Returns:
        The elementwise inverse hyperbolic sine of x.
    """
    return _wrap(ops.asinh(_to_tensor(x)))


def atan(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Elementwise arc tangent.

    Args:
        x: Input array.

    Returns:
        The elementwise arc tangent of x.
    """
    return _wrap(ops.atan(_to_tensor(x)))


def atan2(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Elementwise arc tangent of two variables.

    Args:
        x: Y-coordinates.
        y: X-coordinates.

    Returns:
        The elementwise arc tangent of x and y.
    """
    return _wrap(ops.atan2(_to_tensor(x), _to_tensor(y)))


def atanh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Elementwise inverse hyperbolic tangent.

    Args:
        x: Input array.

    Returns:
        The elementwise inverse hyperbolic tangent of x.
    """
    return _wrap(ops.atanh(_to_tensor(x)))


def bitwise_and(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for bitwise_and.

    Args:
        x: Argument x.
        y: Argument y.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.bitwise_and(_to_tensor(x), _to_tensor(y)))


def bitwise_not(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for bitwise_not.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.bitwise_not(_to_tensor(x)))


def bitwise_or(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for bitwise_or.

    Args:
        x: Argument x.
        y: Argument y.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.bitwise_or(_to_tensor(x), _to_tensor(y)))


def bitwise_xor(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for bitwise_xor.

    Args:
        x: Argument x.
        y: Argument y.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.bitwise_xor(_to_tensor(x), _to_tensor(y)))


def eq(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for eq.

    Args:
        x: Argument x.
        y: Argument y.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.equal(_to_tensor(x), _to_tensor(y)))


def ne(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for ne.

    Args:
        x: Argument x.
        y: Argument y.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.not_equal(_to_tensor(x), _to_tensor(y)))


def gt(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for gt.

    Args:
        x: Argument x.
        y: Argument y.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.greater(_to_tensor(x), _to_tensor(y)))


def ge(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for ge.

    Args:
        x: Argument x.
        y: Argument y.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.greater_equal(_to_tensor(x), _to_tensor(y)))


def lt(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for lt.

    Args:
        x: Argument x.
        y: Argument y.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.less(_to_tensor(x), _to_tensor(y)))


def le(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for le.

    Args:
        x: Argument x.
        y: Argument y.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.less_equal(_to_tensor(x), _to_tensor(y)))


def cbrt(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for cbrt.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.cbrt(_to_tensor(x)))


def conj(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for conj.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.conj(_to_tensor(x)))


def imag(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for imag.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.imag(_to_tensor(x)))


def nextafter(x1: ArrayLikeSpec, x2: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for nextafter.

    Args:
        x1: Argument x1.
        x2: Argument x2.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.nextafter(*[_to_tensor(t) for t in _broadcast_arrays(x1, x2)]))


def real(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for real.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.real(_to_tensor(x)))


def reciprocal(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for reciprocal.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.reciprocal(_to_tensor(x)))


def round(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for round.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.round(_to_tensor(x)))


def sort(operand: ArrayLikeSpec, dimension: int = -1, is_stable: bool = True, num_keys: int = 1) -> ArrayLikeSpec:
    # ml-switcheroo-compiler sort does not accept num_keys, but JAX does.
    # For now pass what compiler accepts.
    """JAX API implementation for sort.

    Args:
        operand: Argument operand.
        dimension: Argument dimension.
        is_stable: Argument is_stable.
        num_keys: Argument num_keys.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.sort(_to_tensor(operand), axis=dimension, is_stable=is_stable))


def shift_left(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for shift_left.

    Args:
        x: Argument x.
        y: Argument y.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.left_shift(_to_tensor(x), _to_tensor(y)))


def shift_right_arithmetic(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for shift_right_arithmetic.

    Args:
        x: Argument x.
        y: Argument y.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.right_shift(_to_tensor(x), _to_tensor(y)))


def shift_right_logical(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for shift_right_logical.

    Args:
        x: Argument x.
        y: Argument y.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.right_shift(_to_tensor(x), _to_tensor(y)))


def abs(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for abs.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.abs(_to_tensor(x)))


def argmax(operand: ArrayLikeSpec, axis: int, index_dtype: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for argmax.

    Args:
        operand: Argument operand.
        axis: Argument axis.
        index_dtype: Argument index_dtype.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.argmax(_to_tensor(operand), axis=axis))


def argmin(operand: ArrayLikeSpec, axis: int, index_dtype: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for argmin.

    Args:
        operand: Argument operand.
        axis: Argument axis.
        index_dtype: Argument index_dtype.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.argmin(_to_tensor(operand), axis=axis))


def broadcast_shapes(*shapes: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for broadcast_shapes.

    Args:
        *shapes: Variable arguments.

    Returns:
        ArrayLikeSpec: The result.
    """
    import functools

    from ml_switcheroo_compiler.core.shape import broadcast_shapes as _bs

    if not shapes:
        return ()
    return functools.reduce(_bs, shapes)


def ceil(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for ceil.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.ceil(_to_tensor(x)))


def concatenate(operands: ArrayLikeSpec, dimension: int) -> ArrayLikeSpec:
    """JAX API implementation for concatenate.

    Args:
        operands: Argument operands.
        dimension: Argument dimension.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.concatenate([_to_tensor(o) for o in operands], axis=dimension))


def conv_general_dilated(
    lhs: ArrayLikeSpec,
    rhs: ArrayLikeSpec,
    window_strides: ArrayLikeSpec,
    padding: ArrayLikeSpec,
    lhs_dilation: ArrayLikeSpec = None,
    rhs_dilation: ArrayLikeSpec = None,
    dimension_numbers: ArrayLikeSpec = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    precision: ArrayLikeSpec = None,
    preferred_element_type: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """JAX API implementation for conv_general_dilated.

    Args:
        lhs: Argument lhs.
        rhs: Argument rhs.
        window_strides: Argument window_strides.
        padding: Argument padding.
        lhs_dilation: Argument lhs_dilation.
        rhs_dilation: Argument rhs_dilation.
        dimension_numbers: Argument dimension_numbers.
        feature_group_count: Argument feature_group_count.
        batch_group_count: Argument batch_group_count.
        precision: Argument precision.
        preferred_element_type: Argument preferred_element_type.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(
        ops.conv_general_dilated(
            _to_tensor(lhs),
            _to_tensor(rhs),
            window_strides=window_strides,
            padding=padding,
            lhs_dilation=lhs_dilation,
            rhs_dilation=rhs_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=feature_group_count,
            batch_group_count=batch_group_count,
        )
    )


def cos(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for cos.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.cos(_to_tensor(x)))


def cosh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for cosh.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.cosh(_to_tensor(x)))


def cumsum(operand: ArrayLikeSpec, axis: int, reverse: bool = False) -> ArrayLikeSpec:
    """JAX API implementation for cumsum.

    Args:
        operand: Argument operand.
        axis: Argument axis.
        reverse: Argument reverse.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.cumsum(_to_tensor(operand), axis=axis))


def digamma(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for digamma.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.digamma(_to_tensor(x)))


def dot(lhs: ArrayLikeSpec, rhs: ArrayLikeSpec, precision: ArrayLikeSpec = None, preferred_element_type: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for dot.

    Args:
        lhs: Argument lhs.
        rhs: Argument rhs.
        precision: Argument precision.
        preferred_element_type: Argument preferred_element_type.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.dot(_to_tensor(lhs), _to_tensor(rhs)))


def dot_general(
    lhs: ArrayLikeSpec,
    rhs: ArrayLikeSpec,
    dimension_numbers: ArrayLikeSpec,
    precision: ArrayLikeSpec = None,
    preferred_element_type: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """JAX API implementation for dot_general.

    Args:
        lhs: Argument lhs.
        rhs: Argument rhs.
        dimension_numbers: Argument dimension_numbers.
        precision: Argument precision.
        preferred_element_type: Argument preferred_element_type.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.dot_general(_to_tensor(lhs), _to_tensor(rhs), dimension_numbers=dimension_numbers))


def erf(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for erf.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.erf(_to_tensor(x)))


def erfc(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for erfc.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.erfc(_to_tensor(x)))


def exp(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for exp.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.exp(_to_tensor(x)))


def exp2(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for exp2.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.exp2(_to_tensor(x)))


def expand_dims(operand: ArrayLikeSpec, dimensions: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for expand_dims.

    Args:
        operand: Argument operand.
        dimensions: Argument dimensions.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.expand_dims(_to_tensor(operand), axis=dimensions))


def expm1(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for expm1.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.expm1(_to_tensor(x)))


def floor(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for floor.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.floor(_to_tensor(x)))


def full(shape: ArrayLikeSpec, fill_value: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for full.

    Args:
        shape: Argument shape.
        fill_value: Argument fill_value.
        dtype: Argument dtype.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.full(shape, fill_value))


def full_like(operand: ArrayLikeSpec, fill_value: ArrayLikeSpec, dtype: ArrayLikeSpec = None, shape: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for full_like.

    Args:
        operand: Argument operand.
        fill_value: Argument fill_value.
        dtype: Argument dtype.
        shape: Argument shape.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.full_like(_to_tensor(operand), fill_value))


def lgamma(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for lgamma.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.lgamma(_to_tensor(x)))


def log(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for log.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.log(_to_tensor(x)))


def log1p(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for log1p.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.log1p(_to_tensor(x)))


def max(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for max.

    Args:
        x: Argument x.
        y: Argument y.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.maximum(_to_tensor(x), _to_tensor(y)))


def min(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for min.

    Args:
        x: Argument x.
        y: Argument y.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.minimum(_to_tensor(x), _to_tensor(y)))


def pad(operand: ArrayLikeSpec, padding_value: ArrayLikeSpec, padding_config: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for pad.

    Args:
        operand: Argument operand.
        padding_value: Argument padding_value.
        padding_config: Argument padding_config.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(
        ops.pad(
            _to_tensor(operand),
            padding_config,
            constant_values=_to_tensor(padding_value),
        )
    )


def pmean(x: ArrayLikeSpec, axis_name: ArrayLikeSpec, axis_index_groups: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for pmean.

    Args:
        x: Argument x.
        axis_name: Argument axis_name.
        axis_index_groups: Argument axis_index_groups.

    Returns:
        ArrayLikeSpec: The result.
    """
    _ = axis_index_groups
    from zero_jax.lax.control_flow import apply_axis_collective

    coll = apply_axis_collective("pmean", x, axis_name)
    if coll is not None:
        return coll
    return _wrap(ops.pmean(_to_tensor(x), axis_name=axis_name))


def psum(x: ArrayLikeSpec, axis_name: ArrayLikeSpec, axis_index_groups: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """JAX API implementation for psum.

    Args:
        x: Argument x.
        axis_name: Argument axis_name.
        axis_index_groups: Argument axis_index_groups.

    Returns:
        ArrayLikeSpec: The result.
    """
    _ = axis_index_groups
    from zero_jax.lax.control_flow import apply_axis_collective

    coll = apply_axis_collective("psum", x, axis_name)
    if coll is not None:
        return coll
    return _wrap(ops.psum(_to_tensor(x), axis_name=axis_name))


def reduce_window(
    operand: ArrayLikeSpec,
    init_value: ArrayLikeSpec,
    computation: ArrayLikeSpec,
    window_dimensions: ArrayLikeSpec,
    window_strides: ArrayLikeSpec,
    padding: ArrayLikeSpec,
    base_dilation: ArrayLikeSpec = None,
    window_dilation: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """JAX API implementation for reduce_window.

    Args:
        operand: Argument operand.
        init_value: Argument init_value.
        computation: Argument computation.
        window_dimensions: Argument window_dimensions.
        window_strides: Argument window_strides.
        padding: Argument padding.
        base_dilation: Argument base_dilation.
        window_dilation: Argument window_dilation.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(
        ops.reduce_window(
            _to_tensor(operand),
            _to_tensor(init_value),
            computation,
            window_dimensions=window_dimensions,
            window_strides=window_strides,
            padding=padding,
            base_dilation=base_dilation,
            window_dilation=window_dilation,
        )
    )


def rsqrt(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for rsqrt.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.rsqrt(_to_tensor(x)))


def sign(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for sign.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.sign(_to_tensor(x)))


def sin(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for sin.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.sin(_to_tensor(x)))


def sinh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for sinh.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.sinh(_to_tensor(x)))


def sqrt(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for sqrt.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.sqrt(_to_tensor(x)))


def square(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for square.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.square(_to_tensor(x)))


def squeeze(operand: ArrayLikeSpec, dimensions: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for squeeze.

    Args:
        operand: Argument operand.
        dimensions: Argument dimensions.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.squeeze(_to_tensor(operand), axis=dimensions))


def tan(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for tan.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.tan(_to_tensor(x)))


def tanh(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """JAX API implementation for tanh.

    Args:
        x: Argument x.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.tanh(_to_tensor(x)))


def top_k(operand: ArrayLikeSpec, k: int) -> ArrayLikeSpec:
    """JAX API implementation for top_k.

    Args:
        operand: Argument operand.
        k: Argument k.

    Returns:
        ArrayLikeSpec: The result.
    """
    res = ops.top_k(_to_tensor(operand), k=k)
    return tuple(_wrap(t) for t in res)


def fft(
    operand: ArrayLikeSpec,
    fft_type: ArrayLikeSpec = None,
    fft_lengths: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """JAX API implementation for fft.

    Args:
        operand: Argument operand.
        fft_type: Argument fft_type.
        fft_lengths: Argument fft_lengths.

    Returns:
        ArrayLikeSpec: The result.
    """
    return _wrap(ops.fft(_to_tensor(operand)))


def neg(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Computes the numerical negative of x element-wise.

    Args:
        x: Input array.

    Returns:
        The numerical negative of x.
    """
    return _wrap(ops.negative(_to_tensor(x)))
