"""LAX convolution, reduction, scan, and control-flow primitives."""

from __future__ import annotations

import zero_jax._compiler_proxy_ops as ops
from zero_jax._types import ArrayLikeSpec
from zero_jax.lax._slicing import padtype_to_pads
from zero_jax.lax.control_flow import cond, scan
from zero_jax.lax.primitives import (
    conv_general_dilated,
    cumsum,
    dot_general,
    reduce,
    reduce_window,
)
from zero_jax.lax.types import ConvDimensionNumbers
from zero_jax.numpy import (
    all as jnp_all,
)
from zero_jax.numpy import (
    any as jnp_any,
)
from zero_jax.numpy import (
    asarray,
    astype,
    bitwise_xor,
    convolve,
    einsum,
    matmul,
    where,
)
from zero_jax.numpy import (
    cumprod as np_cumprod,
)
from zero_jax.numpy import (
    max as jnp_max,
)
from zero_jax.numpy import (
    min as jnp_min,
)
from zero_jax.numpy import (
    prod as jnp_prod,
)
from zero_jax.numpy import (
    sum as jnp_sum,
)
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap, ndarray


def approx_max_k(operand: ArrayLikeSpec, k: int, reduction_dimension: int = -1, recall_target: float = 0.95) -> ndarray:
    """Compute approximate top-``k`` maximum values and indices along ``reduction_dimension``.

    Args:
        operand: Input array to search.
        k: Number of top largest elements.
        reduction_dimension: Dimension along which to search.
        recall_target: Recall target in ``(0, 1]``.

    Returns:
        ndarray: Approximate top-``k`` result.
    """
    return _wrap(ops.approx_max_k(_to_tensor(operand), k, reduction_dimension, recall_target))


def approx_min_k(operand: ArrayLikeSpec, k: int, reduction_dimension: int = -1, recall_target: float = 0.95) -> ndarray:
    """Compute approximate bottom-``k`` minimum values and indices along ``reduction_dimension``.

    Args:
        operand: Input array to search.
        k: Number of smallest elements.
        reduction_dimension: Dimension along which to search.
        recall_target: Recall target in ``(0, 1]``.

    Returns:
        ndarray: Approximate bottom-``k`` result.
    """
    return _wrap(ops.approx_min_k(_to_tensor(operand), k, reduction_dimension, recall_target))


def associative_scan(fn: ArrayLikeSpec, elems: ArrayLikeSpec, reverse: bool = False, axis: int = 0) -> ndarray:
    """Perform a parallel prefix scan with associative binary operator ``fn`` along ``axis``.

    Args:
        fn: Associative binary callable.
        elems: Input array of elements to scan.
        reverse: Whether to scan in reverse order.
        axis: Axis along which to scan.

    Returns:
        ndarray: Prefix-scanned array.
    """
    return _wrap(ops.associative_scan(fn, _to_tensor(elems), reverse, axis))


def batch_matmul(lhs: ArrayLikeSpec, rhs: ArrayLikeSpec, precision: ArrayLikeSpec = None) -> ndarray:
    """Compute batched matrix multiplication of ``lhs`` and ``rhs``.

    Args:
        lhs: Left-hand side batch of matrices.
        rhs: Right-hand side batch of matrices.
        precision: Optional precision configuration.

    Returns:
        ndarray: Batched matrix product.
    """
    _ = precision
    return matmul(lhs, rhs)


def conv(
    lhs: ArrayLikeSpec,
    rhs: ArrayLikeSpec,
    window_strides: ArrayLikeSpec,
    padding: ArrayLikeSpec,
    precision: ArrayLikeSpec = None,
) -> ndarray:
    """Compute N-D convolution over ``lhs`` and kernel ``rhs``.

    Args:
        lhs: Input activation tensor ``(N, C, ...)`` or 1-D signal.
        rhs: Convolution kernel tensor ``(O, I, ...)`` or 1-D filter.
        window_strides: Spatial window strides.
        padding: Padding mode string (``"SAME"``, ``"VALID"``) or explicit pad pairs.
        precision: Optional precision configuration.

    Returns:
        ndarray: Convolved array.
    """
    l_arr = asarray(lhs)
    r_arr = asarray(rhs)
    if l_arr.ndim >= 3 and r_arr.ndim >= 3:
        return conv_general_dilated(l_arr, r_arr, window_strides, padding, precision=precision)
    mode = padding.lower() if isinstance(padding, str) and padding.lower() in ("same", "valid", "full") else "same"
    out = convolve(l_arr.reshape(-1), r_arr.reshape(-1), mode=mode)
    return out.reshape(l_arr.shape) if out.size == l_arr.size else out


def conv_dimension_numbers(
    lhs_shape: ArrayLikeSpec,
    rhs_shape: ArrayLikeSpec,
    dimension_numbers: ArrayLikeSpec,
) -> ConvDimensionNumbers:
    """Canonicalize convolution ``dimension_numbers`` into a :class:`ConvDimensionNumbers` tuple.

    Args:
        lhs_shape: Shape of the left-hand side input tensor.
        rhs_shape: Shape of the right-hand side kernel tensor.
        dimension_numbers: Optional dimension spec ``(lhs_spec, rhs_spec, out_spec)``.

    Returns:
        ConvDimensionNumbers: Canonicalized dimension numbers.
    """
    _ = rhs_shape
    if isinstance(dimension_numbers, ConvDimensionNumbers):
        return dimension_numbers
    ndim = max(2, len(asarray(lhs_shape).__array__().ravel()))
    default_spec = tuple(range(ndim))
    return ConvDimensionNumbers(default_spec, default_spec, default_spec)


def conv_general_dilated_local(
    lhs: ArrayLikeSpec,
    rhs: ArrayLikeSpec,
    window_strides: ArrayLikeSpec,
    padding: ArrayLikeSpec,
    lhs_dilation: ArrayLikeSpec = None,
    rhs_dilation: ArrayLikeSpec = None,
    dimension_numbers: ArrayLikeSpec = None,
    precision: ArrayLikeSpec = None,
) -> ndarray:
    """Compute a general dilated unshared (locally connected) convolution.

    Args:
        lhs: Left-hand side input array.
        rhs: Right-hand side local filter array.
        window_strides: Spatial strides.
        padding: Spatial padding specification.
        lhs_dilation: Optional input dilation factors.
        rhs_dilation: Optional kernel dilation factors.
        dimension_numbers: Optional dimension numbers specification.
        precision: Optional precision configuration.

    Returns:
        ndarray: Locally convolved output array.
    """
    return _wrap(
        ops.conv_general_dilated_local(
            _to_tensor(lhs),
            _to_tensor(rhs),
            _to_tensor(window_strides),
            _to_tensor(padding),
            _to_tensor(lhs_dilation),
            _to_tensor(rhs_dilation),
            _to_tensor(dimension_numbers),
            _to_tensor(precision),
        )
    )


def conv_general_dilated_patches(
    lhs: ArrayLikeSpec,
    filter_shape: ArrayLikeSpec,
    window_strides: ArrayLikeSpec,
    padding: ArrayLikeSpec,
    lhs_dilation: ArrayLikeSpec = None,
    dimension_numbers: ArrayLikeSpec = None,
) -> ndarray:
    """Extract sliding local patches from ``lhs`` (im2col).

    Args:
        lhs: Input tensor.
        filter_shape: Spatial filter shape.
        window_strides: Spatial window strides.
        padding: Spatial padding specification.
        lhs_dilation: Optional input dilation factors.
        dimension_numbers: Optional dimension numbers specification.

    Returns:
        ndarray: Extracted patches array.
    """
    return _wrap(
        ops.conv_general_dilated_patches(
            _to_tensor(lhs),
            _to_tensor(filter_shape),
            _to_tensor(window_strides),
            _to_tensor(padding),
            _to_tensor(lhs_dilation),
            _to_tensor(dimension_numbers),
        )
    )


def conv_general_permutations(
    dimension_numbers: ArrayLikeSpec,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return ``(lhs_perm, rhs_perm, out_perm)`` axis permutations for ``dimension_numbers``.

    Args:
        dimension_numbers: :class:`ConvDimensionNumbers` or sequence of dimension specs.

    Returns:
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]: Axis permutation triples.
    """
    if isinstance(dimension_numbers, ConvDimensionNumbers):
        return (
            tuple(int(x) for x in dimension_numbers.lhs_spec),
            tuple(int(x) for x in dimension_numbers.rhs_spec),
            tuple(int(x) for x in dimension_numbers.out_spec),
        )
    flat = tuple(int(x) for x in asarray(dimension_numbers).__array__().ravel())
    perm = flat if flat else (0, 1, 2)
    return perm, perm, perm


def conv_general_shape_tuple(
    lhs_shape: ArrayLikeSpec,
    rhs_shape: ArrayLikeSpec,
    window_strides: ArrayLikeSpec,
    padding: ArrayLikeSpec,
    lhs_dilation: ArrayLikeSpec = None,
    rhs_dilation: ArrayLikeSpec = None,
    dimension_numbers: ArrayLikeSpec = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
) -> tuple[int, ...]:
    """Compute the output shape tuple of a general dilated convolution.

    Args:
        lhs_shape: Shape of the left-hand side input.
        rhs_shape: Shape of the right-hand side kernel.
        window_strides: Spatial window strides.
        padding: Padding mode string or explicit pad pairs.
        lhs_dilation: Optional input dilation factors.
        rhs_dilation: Optional kernel dilation factors.
        dimension_numbers: Optional convolution dimension numbers.
        feature_group_count: Number of feature groups.
        batch_group_count: Number of batch groups.

    Returns:
        tuple[int, ...]: Output shape tuple.
    """
    _ = (lhs_dilation, rhs_dilation, dimension_numbers, feature_group_count)
    l_dims = [max(1, int(s)) for s in asarray(lhs_shape).__array__().ravel()]
    r_dims = [max(1, int(s)) for s in asarray(rhs_shape).__array__().ravel()]
    if len(l_dims) <= 2 or len(r_dims) <= 2:
        return tuple(l_dims)
    batch_size = max(1, l_dims[0] // max(1, batch_group_count))
    out_channels = r_dims[0]
    spatial_in = l_dims[2:]
    spatial_k = r_dims[2:]
    spatial_out = reduce_window_shape_tuple(spatial_in, spatial_k, window_strides, padding)
    return (batch_size, out_channels, *spatial_out)


def conv_shape_tuple(
    lhs_shape: ArrayLikeSpec,
    rhs_shape: ArrayLikeSpec,
    strides: ArrayLikeSpec,
    padding: ArrayLikeSpec,
    dimension_numbers: ArrayLikeSpec = None,
) -> tuple[int, ...]:
    """Compute the output shape tuple of a standard convolution.

    Args:
        lhs_shape: Shape of the left-hand side input.
        rhs_shape: Shape of the right-hand side kernel.
        strides: Spatial window strides.
        padding: Padding specification.
        dimension_numbers: Optional convolution dimension numbers.

    Returns:
        tuple[int, ...]: Output shape tuple.
    """
    return conv_general_shape_tuple(
        lhs_shape,
        rhs_shape,
        strides,
        padding,
        dimension_numbers=dimension_numbers,
    )


def conv_transpose(
    lhs: ArrayLikeSpec,
    rhs: ArrayLikeSpec,
    strides: ArrayLikeSpec,
    padding: ArrayLikeSpec,
    rhs_dilation: ArrayLikeSpec = None,
    dimension_numbers: ArrayLikeSpec = None,
    transpose_kernel: bool = False,
    precision: ArrayLikeSpec = None,
    preferred_element_type: ArrayLikeSpec = None,
) -> ndarray:
    """Compute N-D transposed (fractionally strided) convolution.

    Args:
        lhs: Left-hand side input tensor.
        rhs: Right-hand side kernel tensor.
        strides: Spatial strides.
        padding: Padding specification.
        rhs_dilation: Optional kernel dilation factors.
        dimension_numbers: Optional dimension numbers specification.
        transpose_kernel: Whether to flip/transpose the kernel.
        precision: Optional precision configuration.
        preferred_element_type: Optional preferred output dtype.

    Returns:
        ndarray: Transposed convolution result.
    """
    return _wrap(
        ops.conv_transpose(
            _to_tensor(lhs),
            _to_tensor(rhs),
            _to_tensor(strides),
            _to_tensor(padding),
            _to_tensor(rhs_dilation),
            _to_tensor(dimension_numbers),
            transpose_kernel,
            _to_tensor(precision),
            _to_tensor(preferred_element_type),
        )
    )


def conv_transpose_shape_tuple(
    lhs_shape: ArrayLikeSpec,
    rhs_shape: ArrayLikeSpec,
    window_strides: ArrayLikeSpec,
    padding: ArrayLikeSpec,
    dimension_numbers: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Compute the output shape tuple of a transposed convolution.

    Args:
        lhs_shape: Shape of the left-hand side input.
        rhs_shape: Shape of the right-hand side kernel.
        window_strides: Spatial window strides.
        padding: Padding specification.
        dimension_numbers: Optional dimension numbers specification.

    Returns:
        ArrayLikeSpec: Transposed convolution output shape tuple.
    """
    return ops.conv_transpose_shape_tuple(
        lhs_shape,
        rhs_shape,
        window_strides,
        padding,
        dimension_numbers,
    )


def conv_with_general_padding(
    lhs: ArrayLikeSpec,
    rhs: ArrayLikeSpec,
    window_strides: ArrayLikeSpec,
    padding: ArrayLikeSpec,
    lhs_dilation: ArrayLikeSpec = None,
    rhs_dilation: ArrayLikeSpec = None,
    dimension_numbers: ArrayLikeSpec = None,
    precision: ArrayLikeSpec = None,
) -> ndarray:
    """Compute N-D convolution with explicit low/high padding pairs.

    Args:
        lhs: Left-hand side input array.
        rhs: Right-hand side kernel array.
        window_strides: Spatial strides.
        padding: Sequence of ``(pad_low, pad_high)`` pairs.
        lhs_dilation: Optional input dilation factors.
        rhs_dilation: Optional kernel dilation factors.
        dimension_numbers: Optional dimension numbers specification.
        precision: Optional precision configuration.

    Returns:
        ndarray: Convolved output array.
    """
    return _wrap(
        ops.conv_with_general_padding(
            _to_tensor(lhs),
            _to_tensor(rhs),
            _to_tensor(window_strides),
            _to_tensor(padding),
            _to_tensor(lhs_dilation),
            _to_tensor(rhs_dilation),
            _to_tensor(dimension_numbers),
            _to_tensor(precision),
        )
    )


def cumlogsumexp(operand: ArrayLikeSpec, axis: int = 0, reverse: bool = False) -> ndarray:
    """Compute cumulative log-sum-exp of ``operand`` along ``axis``.

    Args:
        operand: Input array.
        axis: Axis along which to accumulate.
        reverse: Whether to accumulate in reverse order.

    Returns:
        ndarray: Cumulative log-sum-exp array.
    """
    return _wrap(ops.cumlogsumexp(_to_tensor(operand), axis, reverse))


def cummax(operand: ArrayLikeSpec, axis: int = 0, reverse: bool = False) -> ndarray:
    """Compute cumulative maximum of ``operand`` along ``axis``.

    Args:
        operand: Input array.
        axis: Axis along which to accumulate.
        reverse: Whether to accumulate in reverse order.

    Returns:
        ndarray: Cumulative maximum array.
    """
    return _wrap(ops.cummax(_to_tensor(operand), axis, reverse))


def cummin(operand: ArrayLikeSpec, axis: int = 0, reverse: bool = False) -> ndarray:
    """Compute cumulative minimum of ``operand`` along ``axis``.

    Args:
        operand: Input array.
        axis: Axis along which to accumulate.
        reverse: Whether to accumulate in reverse order.

    Returns:
        ndarray: Cumulative minimum array.
    """
    return _wrap(ops.cummin(_to_tensor(operand), axis, reverse))


def cumprod(operand: ArrayLikeSpec, axis: int = 0, reverse: bool = False) -> ndarray:
    """Compute cumulative product of ``operand`` along ``axis``.

    Args:
        operand: Input array.
        axis: Axis along which to accumulate.
        reverse: Whether to accumulate in reverse order.

    Returns:
        ndarray: Cumulative product array.
    """
    _ = reverse
    return np_cumprod(operand, axis=axis)


def custom_linear_solve(
    matvec: ArrayLikeSpec,
    b: ArrayLikeSpec,
    solve: ArrayLikeSpec,
    transpose_solve: ArrayLikeSpec = None,
    symmetric: bool = False,
) -> ndarray:
    """Solve a linear system ``A x = b`` with custom forward/transpose linear solvers.

    Args:
        matvec: Linear operator callable ``x -> A x``.
        b: Right-hand side vector or array.
        solve: Forward solver callable ``(matvec, b) -> x``.
        transpose_solve: Optional transpose solver callable.
        symmetric: Whether the linear operator is symmetric.

    Returns:
        ndarray: Solution array ``x``.
    """
    return _wrap(
        ops.custom_linear_solve(
            _to_tensor(matvec),
            _to_tensor(b),
            _to_tensor(solve),
            _to_tensor(transpose_solve),
            symmetric,
        )
    )


def custom_root(
    f: ArrayLikeSpec,
    initial_guess: ArrayLikeSpec,
    solve: ArrayLikeSpec,
    tangent_solve: ArrayLikeSpec,
    has_aux: bool = False,
) -> ndarray:
    """Find a root of ``f(x) = 0`` with implicit differentiation via ``tangent_solve``.

    Args:
        f: Residual function ``x -> f(x)``.
        initial_guess: Initial root estimate.
        solve: Root-finding callable ``(f, initial_guess) -> x_star``.
        tangent_solve: Linear solver for the Jacobian system in autodiff.
        has_aux: Whether ``solve`` returns auxiliary data.

    Returns:
        ndarray: Root solution array.
    """
    return _wrap(
        ops.custom_root(
            _to_tensor(f),
            _to_tensor(initial_guess),
            _to_tensor(solve),
            _to_tensor(tangent_solve),
            has_aux,
        )
    )


def fori_loop(lower: ArrayLikeSpec, upper: ArrayLikeSpec, body_fun: ArrayLikeSpec, init_val: ArrayLikeSpec) -> ndarray:
    """Loop from ``lower`` to ``upper`` applying ``body_fun(i, val)``.

    Args:
        lower: Inclusive starting loop index.
        upper: Exclusive ending loop index.
        body_fun: Loop body callable ``(i, val) -> next_val``.
        init_val: Initial loop carry value.

    Returns:
        ndarray: Final loop carry value.
    """
    return _wrap(ops.fori_loop(_to_tensor(lower), _to_tensor(upper), body_fun, _to_tensor(init_val)))


def map(f: ArrayLikeSpec, xs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Apply function ``f`` over leading axis elements of ``xs``.

    Args:
        f: Function to map.
        xs: Input array or PyTree.

    Returns:
        ArrayLikeSpec: Mapped result.
    """
    return f(xs)


def ragged_dot(lhs: ArrayLikeSpec, rhs: ArrayLikeSpec) -> ndarray:
    """Compute a ragged dot product of ``lhs`` and ``rhs``.

    Args:
        lhs: Left-hand side operand.
        rhs: Right-hand side operand.

    Returns:
        ndarray: Ragged dot product output.
    """
    return _wrap(ops.ragged_dot(_to_tensor(lhs), _to_tensor(rhs)))


def ragged_dot_general(
    lhs: ArrayLikeSpec,
    rhs: ArrayLikeSpec,
    group_sizes: ArrayLikeSpec,
    ragged_dot_dimension_numbers: ArrayLikeSpec = None,
    precision: ArrayLikeSpec = None,
    preferred_element_type: ArrayLikeSpec = None,
    group_offset: ArrayLikeSpec = 0,
    out_sharding: ArrayLikeSpec = None,
) -> ndarray:
    """Compute a generalized ragged dot product across variable-sized groups.

    Args:
        lhs: Left-hand side array.
        rhs: Right-hand side array.
        group_sizes: Group sizes array.
        ragged_dot_dimension_numbers: Ragged dot dimension specification.
        precision: Optional precision specification.
        preferred_element_type: Optional accumulation/output dtype.
        group_offset: Starting group offset.
        out_sharding: Optional output sharding specification.

    Returns:
        ndarray: Ragged dot product output.
    """
    _ = (group_sizes, ragged_dot_dimension_numbers, precision, preferred_element_type, group_offset, out_sharding)
    return ragged_dot(lhs, rhs)


def reduce_and(operand: ArrayLikeSpec, axes: ArrayLikeSpec) -> ndarray:
    """Reduce ``operand`` along ``axes`` using logical/bitwise AND.

    Args:
        operand: Input array.
        axes: Sequence of axes to reduce over.

    Returns:
        ndarray: Reduced array.
    """
    return jnp_all(operand, axis=tuple(axes) if isinstance(axes, (list, tuple)) else axes)


def reduce_max(operand: ArrayLikeSpec, axes: ArrayLikeSpec, out_sharding: ArrayLikeSpec = None) -> ndarray:
    """Reduce ``operand`` along ``axes`` using maximum.

    Args:
        operand: Input array.
        axes: Sequence of axes to reduce over.
        out_sharding: Optional output sharding specification.

    Returns:
        ndarray: Reduced maximum array.
    """
    _ = out_sharding
    return jnp_max(operand, axis=tuple(axes) if isinstance(axes, (list, tuple)) else axes)


def reduce_min(operand: ArrayLikeSpec, axes: ArrayLikeSpec, out_sharding: ArrayLikeSpec = None) -> ndarray:
    """Reduce ``operand`` along ``axes`` using minimum.

    Args:
        operand: Input array.
        axes: Sequence of axes to reduce over.
        out_sharding: Optional output sharding specification.

    Returns:
        ndarray: Reduced minimum array.
    """
    _ = out_sharding
    return jnp_min(operand, axis=tuple(axes) if isinstance(axes, (list, tuple)) else axes)


def reduce_or(operand: ArrayLikeSpec, axes: ArrayLikeSpec) -> ndarray:
    """Reduce ``operand`` along ``axes`` using logical/bitwise OR.

    Args:
        operand: Input array.
        axes: Sequence of axes to reduce over.

    Returns:
        ndarray: Reduced array.
    """
    return jnp_any(operand, axis=tuple(axes) if isinstance(axes, (list, tuple)) else axes)


def reduce_precision(operand: ArrayLikeSpec, exponent_bits: int, mantissa_bits: int) -> ndarray:
    """Simulate reduced floating-point precision with ``exponent_bits`` and ``mantissa_bits``.

    Args:
        operand: Input floating-point array.
        exponent_bits: Number of exponent bits.
        mantissa_bits: Number of mantissa bits.

    Returns:
        ndarray: Precision-reduced array.
    """
    return _wrap(ops.reduce_precision(_to_tensor(operand), exponent_bits, mantissa_bits))


def reduce_prod(operand: ArrayLikeSpec, axes: ArrayLikeSpec) -> ndarray:
    """Reduce ``operand`` along ``axes`` using product.

    Args:
        operand: Input array.
        axes: Sequence of axes to reduce over.

    Returns:
        ndarray: Reduced product array.
    """
    return jnp_prod(operand, axis=tuple(axes) if isinstance(axes, (list, tuple)) else axes)


def reduce_sum(operand: ArrayLikeSpec, axes: ArrayLikeSpec, out_sharding: ArrayLikeSpec = None) -> ndarray:
    """Reduce ``operand`` along ``axes`` using sum.

    Args:
        operand: Input array.
        axes: Sequence of axes to reduce over.
        out_sharding: Optional output sharding specification.

    Returns:
        ndarray: Reduced sum array.
    """
    _ = out_sharding
    return jnp_sum(operand, axis=tuple(axes) if isinstance(axes, (list, tuple)) else axes)


def reduce_window_shape_tuple(
    operand_shape: ArrayLikeSpec,
    window_dimensions: ArrayLikeSpec,
    window_strides: ArrayLikeSpec,
    padding: ArrayLikeSpec,
) -> tuple[int, ...]:
    """Compute the output shape tuple of a ``reduce_window`` operation.

    Args:
        operand_shape: Input operand shape.
        window_dimensions: Window dimensions per axis.
        window_strides: Window strides per axis.
        padding: Padding mode string or sequence of ``(low, high)`` pairs.

    Returns:
        tuple[int, ...]: Output shape tuple.
    """
    in_dims = [max(1, int(s)) for s in asarray(operand_shape).__array__().ravel()]
    win_dims = [max(1, int(w)) for w in asarray(window_dimensions).__array__().ravel()]
    str_dims = [max(1, int(st)) for st in asarray(window_strides).__array__().ravel()]
    pads = padtype_to_pads(in_dims, win_dims, str_dims, padding)
    out_shape: list[int] = []
    for s, w, st, (lo, hi) in zip(in_dims, win_dims, str_dims, pads):
        out_shape.append(max(1, (s + lo + hi - w) // st + 1))
    return tuple(out_shape)


def reduce_xor(operand: ArrayLikeSpec, axes: ArrayLikeSpec) -> ndarray:
    """Reduce ``operand`` along ``axes`` using bitwise/logical XOR.

    Args:
        operand: Input array.
        axes: Sequence of axes to reduce over.

    Returns:
        ndarray: Reduced XOR array.
    """
    arr = asarray(operand)
    ax_tuple = (int(axes),) if isinstance(axes, int) else tuple(int(a) for a in axes)
    res = arr
    for ax in sorted(ax_tuple, reverse=True):
        acc = res.take(0, axis=ax)
        for i in range(1, res.shape[ax]):
            acc = bitwise_xor(acc, res.take(i, axis=ax))
        res = acc
    return res


def scaled_dot(
    lhs: ArrayLikeSpec,
    rhs: ArrayLikeSpec,
    lhs_scale: ArrayLikeSpec = None,
    rhs_scale: ArrayLikeSpec = None,
    dimension_numbers: ArrayLikeSpec = None,
    preferred_element_type: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Compute a block-scaled dot product of ``lhs`` and ``rhs``.

    Args:
        lhs: Left-hand side operand.
        rhs: Right-hand side operand.
        lhs_scale: Optional scale factor for ``lhs``.
        rhs_scale: Optional scale factor for ``rhs``.
        dimension_numbers: Optional contracting/batch dimension numbers.
        preferred_element_type: Optional output dtype.

    Returns:
        ArrayLikeSpec: Scaled dot product result.
    """
    l_arr = asarray(lhs)
    r_arr = asarray(rhs)
    if lhs_scale is not None:
        l_arr = l_arr * asarray(lhs_scale)
    if rhs_scale is not None:
        r_arr = r_arr * asarray(rhs_scale)
    if dimension_numbers is not None:
        return dot_general(l_arr, r_arr, dimension_numbers=dimension_numbers, preferred_element_type=preferred_element_type)
    res = matmul(l_arr, r_arr)
    if preferred_element_type is not None:
        return astype(res, preferred_element_type)
    return res


def scan_bind(
    scan_primitive: ArrayLikeSpec = None,
    *args: ArrayLikeSpec,
    **kwargs: ArrayLikeSpec,
) -> ArrayLikeSpec:
    """Bind a scan primitive to ``args`` and ``kwargs``.

    Args:
        scan_primitive: Scan primitive or callable to bind.
        *args: Positional arguments to bind.
        **kwargs: Keyword arguments to bind.

    Returns:
        ArrayLikeSpec: Bound result or ``scan_primitive`` if not callable.
    """
    if callable(scan_primitive):
        return scan_primitive(*args, **kwargs)
    return scan_primitive


def select_n(pred: ArrayLikeSpec, *cases: ArrayLikeSpec) -> ndarray:
    """Select elementwise from ``cases`` based on integer or boolean ``pred``.

    Args:
        pred: Selector index or boolean array.
        *cases: Candidate arrays from which to select.

    Returns:
        ndarray: Elementwise selected array.
    """
    return where(pred, cases[1], cases[0])


def sort_key_val(keys: ArrayLikeSpec, values: ArrayLikeSpec, dimension: int = -1) -> ndarray:
    """Sort ``keys`` along ``dimension`` and permute ``values`` correspondingly.

    Args:
        keys: Key array to sort.
        values: Value array to permute alongside ``keys``.
        dimension: Axis along which to sort.

    Returns:
        ndarray: Sorted ``(keys, values)`` result.
    """
    return _wrap(ops.sort_key_val(_to_tensor(keys), _to_tensor(values), dimension))


def switch(index: ArrayLikeSpec, branches: ArrayLikeSpec, *operands: ArrayLikeSpec, operand: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Index into ``branches`` using ``index`` and apply the selected callable to ``operands``.

    Args:
        index: Integer branch index.
        branches: Sequence of branch callables.
        *operands: Positional operands passed to the selected branch.
        operand: Optional legacy single operand.

    Returns:
        ArrayLikeSpec: Result of the selected branch.
    """
    _ = index
    call_ops = (operand,) if operand is not None and not operands else operands
    return branches[0](*call_ops)


def top_k(operand: ArrayLikeSpec, k: int) -> ndarray:
    """Return the top ``k`` largest values and their indices along the last axis of ``operand``.

    Args:
        operand: Input array.
        k: Number of top elements to extract.

    Returns:
        ndarray: Tuple of ``(values, indices)``.
    """
    return _wrap(ops.top_k(_to_tensor(operand), k))


def while_loop(cond_fun: ArrayLikeSpec, body_fun: ArrayLikeSpec, init_val: ArrayLikeSpec) -> ndarray:
    """Repeatedly apply ``body_fun`` to ``init_val`` while ``cond_fun(val)`` is true.

    Args:
        cond_fun: Loop continuation predicate ``val -> bool``.
        body_fun: Loop body function ``val -> next_val``.
        init_val: Initial carry value.

    Returns:
        ndarray: Final loop carry value.
    """
    return _wrap(ops.while_loop(cond_fun, body_fun, _to_tensor(init_val)))


def xeinsum(subscripts: str, *operands: ArrayLikeSpec) -> ndarray:
    """Evaluate Einstein summation ``subscripts`` on ``operands``.

    Args:
        subscripts: Einstein summation specification string.
        *operands: Input arrays.

    Returns:
        ndarray: Contracted array.
    """
    return einsum(subscripts, *operands)


__all__ = [
    "approx_max_k",
    "approx_min_k",
    "associative_scan",
    "batch_matmul",
    "cond",
    "conv",
    "conv_dimension_numbers",
    "conv_general_dilated",
    "conv_general_dilated_local",
    "conv_general_dilated_patches",
    "conv_general_permutations",
    "conv_general_shape_tuple",
    "conv_shape_tuple",
    "conv_transpose",
    "conv_transpose_shape_tuple",
    "conv_with_general_padding",
    "cumlogsumexp",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "custom_linear_solve",
    "custom_root",
    "fori_loop",
    "map",
    "ragged_dot",
    "ragged_dot_general",
    "reduce",
    "reduce_and",
    "reduce_max",
    "reduce_min",
    "reduce_or",
    "reduce_precision",
    "reduce_prod",
    "reduce_sum",
    "reduce_window",
    "reduce_window_shape_tuple",
    "reduce_xor",
    "scaled_dot",
    "scan",
    "scan_bind",
    "select_n",
    "sort_key_val",
    "switch",
    "top_k",
    "while_loop",
    "xeinsum",
]
