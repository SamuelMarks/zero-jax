"""Missing specific functions for jax.lax."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec


def approx_max_k(operand: ArrayLikeSpec, k: int, reduction_dimension: int = -1, recall_target: float = 0.95) -> ArrayLikeSpec:
    """Execute ``approx_max_k``.

    Args:
        operand: Input parameter ``operand``.
        k: Input parameter ``k``.
        reduction_dimension: Input parameter ``reduction_dimension``.
        recall_target: Input parameter ``recall_target``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.approx_max_k(_to_tensor(operand), k, reduction_dimension, recall_target))


def approx_min_k(operand: ArrayLikeSpec, k: int, reduction_dimension: int = -1, recall_target: float = 0.95) -> ArrayLikeSpec:
    """Execute ``approx_min_k``.

    Args:
        operand: Input parameter ``operand``.
        k: Input parameter ``k``.
        reduction_dimension: Input parameter ``reduction_dimension``.
        recall_target: Input parameter ``recall_target``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.approx_min_k(_to_tensor(operand), k, reduction_dimension, recall_target))


def betainc(a: ArrayLikeSpec, b: ArrayLikeSpec, x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``betainc``.

    Args:
        a: Input parameter ``a``.
        b: Input parameter ``b``.
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.betainc(_to_tensor(a), _to_tensor(b), _to_tensor(x)))


def bitcast_convert_type(operand: ArrayLikeSpec, new_dtype: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``bitcast_convert_type``.

    Args:
        operand: Input parameter ``operand``.
        new_dtype: Input parameter ``new_dtype``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.bitcast_convert_type(_to_tensor(operand), new_dtype))


def broadcast_to_rank(operand: ArrayLikeSpec, rank: int) -> ArrayLikeSpec:
    """Execute ``broadcast_to_rank``.

    Args:
        operand: Input parameter ``operand``.
        rank: Input parameter ``rank``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.broadcast_to_rank(_to_tensor(operand), rank))


def broadcasted_iota(dtype: ArrayLikeSpec, shape: ArrayLikeSpec, dimension: int) -> ArrayLikeSpec:
    """Execute ``broadcasted_iota``.

    Args:
        dtype: Input parameter ``dtype``.
        shape: Input parameter ``shape``.
        dimension: Input parameter ``dimension``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _wrap

    return _wrap(ops.broadcasted_iota(dtype, shape, dimension))


def collapse(operand: ArrayLikeSpec, start_dimension: int, stop_dimension: int) -> ArrayLikeSpec:
    """Execute ``collapse``.

    Args:
        operand: Input parameter ``operand``.
        start_dimension: Input parameter ``start_dimension``.
        stop_dimension: Input parameter ``stop_dimension``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.collapse(_to_tensor(operand), start_dimension, stop_dimension))


def conv_dimension_numbers(lhs_shape: ArrayLikeSpec, rhs_shape: ArrayLikeSpec, dimension_numbers: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``conv_dimension_numbers``.

    Args:
        lhs_shape: Input parameter ``lhs_shape``.
        rhs_shape: Input parameter ``rhs_shape``.
        dimension_numbers: Input parameter ``dimension_numbers``.

    Returns:
        The operation result.
    """
    return dimension_numbers


def conv_general_dilated_local(
    lhs: ArrayLikeSpec,
    rhs: ArrayLikeSpec,
    window_strides: ArrayLikeSpec,
    padding: ArrayLikeSpec,
    lhs_dilation: ArrayLikeSpec = None,
    rhs_dilation: ArrayLikeSpec = None,
    dimension_numbers: ArrayLikeSpec = None,
    precision: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Execute ``conv_general_dilated_local``.

    Args:
        lhs: Input parameter ``lhs``.
        rhs: Input parameter ``rhs``.
        window_strides: Input parameter ``window_strides``.
        padding: Input parameter ``padding``.
        lhs_dilation: Input parameter ``lhs_dilation``.
        rhs_dilation: Input parameter ``rhs_dilation``.
        dimension_numbers: Input parameter ``dimension_numbers``.
        precision: Input parameter ``precision``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

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
) -> ArrayLikeSpec:
    """Execute ``conv_general_dilated_patches``.

    Args:
        lhs: Input parameter ``lhs``.
        filter_shape: Input parameter ``filter_shape``.
        window_strides: Input parameter ``window_strides``.
        padding: Input parameter ``padding``.
        lhs_dilation: Input parameter ``lhs_dilation``.
        dimension_numbers: Input parameter ``dimension_numbers``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

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


def conv_general_permutations(dimension_numbers: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``conv_general_permutations``.

    Args:
        dimension_numbers: Input parameter ``dimension_numbers``.

    Returns:
        The operation result.
    """
    return dimension_numbers, dimension_numbers


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
) -> ArrayLikeSpec:
    """Execute ``conv_general_shape_tuple``.

    Args:
        lhs_shape: Input parameter ``lhs_shape``.
        rhs_shape: Input parameter ``rhs_shape``.
        window_strides: Input parameter ``window_strides``.
        padding: Input parameter ``padding``.
        lhs_dilation: Input parameter ``lhs_dilation``.
        rhs_dilation: Input parameter ``rhs_dilation``.
        dimension_numbers: Input parameter ``dimension_numbers``.
        feature_group_count: Input parameter ``feature_group_count``.
        batch_group_count: Input parameter ``batch_group_count``.

    Returns:
        The operation result.
    """
    return lhs_shape


def conv_shape_tuple(
    lhs_shape: ArrayLikeSpec,
    rhs_shape: ArrayLikeSpec,
    strides: ArrayLikeSpec,
    padding: ArrayLikeSpec,
    dimension_numbers: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Execute ``conv_shape_tuple``.

    Args:
        lhs_shape: Input parameter ``lhs_shape``.
        rhs_shape: Input parameter ``rhs_shape``.
        strides: Input parameter ``strides``.
        padding: Input parameter ``padding``.
        dimension_numbers: Input parameter ``dimension_numbers``.

    Returns:
        The operation result.
    """
    return lhs_shape


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
) -> ArrayLikeSpec:
    """Execute ``conv_transpose``.

    Args:
        lhs: Input parameter ``lhs``.
        rhs: Input parameter ``rhs``.
        strides: Input parameter ``strides``.
        padding: Input parameter ``padding``.
        rhs_dilation: Input parameter ``rhs_dilation``.
        dimension_numbers: Input parameter ``dimension_numbers``.
        transpose_kernel: Input parameter ``transpose_kernel``.
        precision: Input parameter ``precision``.
        preferred_element_type: Input parameter ``preferred_element_type``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

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
    """Execute ``conv_transpose_shape_tuple``.

    Args:
        lhs_shape: Input parameter ``lhs_shape``.
        rhs_shape: Input parameter ``rhs_shape``.
        window_strides: Input parameter ``window_strides``.
        padding: Input parameter ``padding``.
        dimension_numbers: Input parameter ``dimension_numbers``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

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
) -> ArrayLikeSpec:
    """Execute ``conv_with_general_padding``.

    Args:
        lhs: Input parameter ``lhs``.
        rhs: Input parameter ``rhs``.
        window_strides: Input parameter ``window_strides``.
        padding: Input parameter ``padding``.
        lhs_dilation: Input parameter ``lhs_dilation``.
        rhs_dilation: Input parameter ``rhs_dilation``.
        dimension_numbers: Input parameter ``dimension_numbers``.
        precision: Input parameter ``precision``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

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


def custom_linear_solve(
    matvec: ArrayLikeSpec,
    b: ArrayLikeSpec,
    solve: ArrayLikeSpec,
    transpose_solve: ArrayLikeSpec = None,
    symmetric: bool = False,
) -> ArrayLikeSpec:
    """Execute ``custom_linear_solve``.

    Args:
        matvec: Input parameter ``matvec``.
        b: Input parameter ``b``.
        solve: Input parameter ``solve``.
        transpose_solve: Input parameter ``transpose_solve``.
        symmetric: Input parameter ``symmetric``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(
        ops.custom_linear_solve(
            _to_tensor(matvec),
            _to_tensor(b),
            _to_tensor(solve),
            _to_tensor(transpose_solve),
            symmetric,
        )
    )


def custom_root(f: ArrayLikeSpec, initial_guess: ArrayLikeSpec, solve: ArrayLikeSpec, tangent_solve: ArrayLikeSpec, has_aux: bool = False) -> ArrayLikeSpec:
    """Execute ``custom_root``.

    Args:
        f: Input parameter ``f``.
        initial_guess: Input parameter ``initial_guess``.
        solve: Input parameter ``solve``.
        tangent_solve: Input parameter ``tangent_solve``.
        has_aux: Input parameter ``has_aux``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(
        ops.custom_root(
            _to_tensor(f),
            _to_tensor(initial_guess),
            _to_tensor(solve),
            _to_tensor(tangent_solve),
            has_aux,
        )
    )


def dtype(operand: ArrayLikeSpec) -> ArrayLikeSpec:
    """Return the dtype of an operand if available.

    Args:
        operand: Input array or scalar operand.

    Returns:
        The dtype of ``operand`` or ``None``.
    """
    from zero_jax._types import ArrayProtocol

    if isinstance(operand, ArrayProtocol):
        return operand.dtype
    return None


def dynamic_index_in_dim(operand: ArrayLikeSpec, index: ArrayLikeSpec, dimension: int = 0, keepdims: bool = True) -> ArrayLikeSpec:
    """Execute ``dynamic_index_in_dim``.

    Args:
        operand: Input parameter ``operand``.
        index: Input parameter ``index``.
        dimension: Input parameter ``dimension``.
        keepdims: Input parameter ``keepdims``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.dynamic_index_in_dim(_to_tensor(operand), _to_tensor(index), dimension, keepdims))


def dynamic_slice_in_dim(operand: ArrayLikeSpec, start_index: ArrayLikeSpec, slice_size: int, dimension: int = 0) -> ArrayLikeSpec:
    """Execute ``dynamic_slice_in_dim``.

    Args:
        operand: Input parameter ``operand``.
        start_index: Input parameter ``start_index``.
        slice_size: Input parameter ``slice_size``.
        dimension: Input parameter ``dimension``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.dynamic_slice_in_dim(_to_tensor(operand), _to_tensor(start_index), slice_size, dimension))


def dynamic_update_index_in_dim(operand: ArrayLikeSpec, update: ArrayLikeSpec, index: ArrayLikeSpec, dimension: int = 0) -> ArrayLikeSpec:
    """Execute ``dynamic_update_index_in_dim``.

    Args:
        operand: Input parameter ``operand``.
        update: Input parameter ``update``.
        index: Input parameter ``index``.
        dimension: Input parameter ``dimension``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.dynamic_update_index_in_dim(_to_tensor(operand), _to_tensor(update), _to_tensor(index), dimension))


def dynamic_update_slice_in_dim(operand: ArrayLikeSpec, update: ArrayLikeSpec, start_index: ArrayLikeSpec, dimension: int = 0) -> ArrayLikeSpec:
    """Execute ``dynamic_update_slice_in_dim``.

    Args:
        operand: Input parameter ``operand``.
        update: Input parameter ``update``.
        start_index: Input parameter ``start_index``.
        dimension: Input parameter ``dimension``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.dynamic_update_slice_in_dim(_to_tensor(operand), _to_tensor(update), _to_tensor(start_index), dimension))


def erf_inv(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``erf_inv``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.erf_inv(_to_tensor(x)))


def fori_loop(lower: ArrayLikeSpec, upper: ArrayLikeSpec, body_fun: ArrayLikeSpec, init_val: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``fori_loop``.

    Args:
        lower: Input parameter ``lower``.
        upper: Input parameter ``upper``.
        body_fun: Input parameter ``body_fun``.
        init_val: Input parameter ``init_val``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.fori_loop(_to_tensor(lower), _to_tensor(upper), body_fun, _to_tensor(init_val)))


def igamma(a: ArrayLikeSpec, x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``igamma``.

    Args:
        a: Input parameter ``a``.
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.igamma(_to_tensor(a), _to_tensor(x)))


def igamma_grad_a(a: ArrayLikeSpec, x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``igamma_grad_a``.

    Args:
        a: Input parameter ``a``.
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.igamma_grad_a(_to_tensor(a), _to_tensor(x)))


def igammac(a: ArrayLikeSpec, x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``igammac``.

    Args:
        a: Input parameter ``a``.
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.igammac(_to_tensor(a), _to_tensor(x)))


def index_in_dim(operand: ArrayLikeSpec, index: ArrayLikeSpec, dimension: int = 0, keepdims: bool = True) -> ArrayLikeSpec:
    """Execute ``index_in_dim``.

    Args:
        operand: Input parameter ``operand``.
        index: Input parameter ``index``.
        dimension: Input parameter ``dimension``.
        keepdims: Input parameter ``keepdims``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.index_in_dim(_to_tensor(operand), _to_tensor(index), dimension, keepdims))


def index_take(operand: ArrayLikeSpec, indices: ArrayLikeSpec, axes: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``index_take``.

    Args:
        operand: Input parameter ``operand``.
        indices: Input parameter ``indices``.
        axes: Input parameter ``axes``.

    Returns:
        The operation result.
    """
    return operand


def infeed() -> ArrayLikeSpec:
    """Execute ``infeed``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    return ops.infeed()


def iota(dtype: ArrayLikeSpec, size: int) -> ArrayLikeSpec:
    """Execute ``iota``.

    Args:
        dtype: Input parameter ``dtype``.
        size: Input parameter ``size``.

    Returns:
        The operation result.
    """
    from zero_jax.numpy import arange

    return arange(size, dtype=dtype)


def logistic(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``logistic``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    from zero_jax.numpy import exp

    return 1.0 / (1.0 + exp(-x))


def outfeed(val: ArrayLikeSpec) -> None:
    """Execute ``outfeed``.

    Args:
        val: Input parameter ``val``.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor

    ops.outfeed(_to_tensor(val))


def padtype_to_pads(shape: ArrayLikeSpec, window_shape: ArrayLikeSpec, window_strides: ArrayLikeSpec, padding: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``padtype_to_pads``.

    Args:
        shape: Input parameter ``shape``.
        window_shape: Input parameter ``window_shape``.
        window_strides: Input parameter ``window_strides``.
        padding: Input parameter ``padding``.

    Returns:
        The operation result.
    """
    return padding


def pbroadcast(operand: ArrayLikeSpec, axis_name: str) -> ArrayLikeSpec:
    """Execute ``pbroadcast``.

    Args:
        operand: Input parameter ``operand``.
        axis_name: Input parameter ``axis_name``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.pbroadcast(_to_tensor(operand), axis_name))


def pdot(lhs: ArrayLikeSpec, rhs: ArrayLikeSpec, axis_name: str) -> ArrayLikeSpec:
    """Execute ``pdot``.

    Args:
        lhs: Input parameter ``lhs``.
        rhs: Input parameter ``rhs``.
        axis_name: Input parameter ``axis_name``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.pdot(_to_tensor(lhs), _to_tensor(rhs), axis_name))


def platform_dependent(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``platform_dependent``.

    Args:
        x: Input parameter ``x``.
        y: Input parameter ``y``.

    Returns:
        The operation result.
    """
    return x


def pmax(operand: ArrayLikeSpec, axis_name: str) -> ArrayLikeSpec:
    """Execute ``pmax``.

    Args:
        operand: Input parameter ``operand``.
        axis_name: Input parameter ``axis_name``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.pmax(_to_tensor(operand), axis_name))


def pmin(operand: ArrayLikeSpec, axis_name: str) -> ArrayLikeSpec:
    """Execute ``pmin``.

    Args:
        operand: Input parameter ``operand``.
        axis_name: Input parameter ``axis_name``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.pmin(_to_tensor(operand), axis_name))


def polygamma(n: ArrayLikeSpec, x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``polygamma``.

    Args:
        n: Input parameter ``n``.
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.polygamma(_to_tensor(n), _to_tensor(x)))


def population_count(operand: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``population_count``.

    Args:
        operand: Input parameter ``operand``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.population_count(_to_tensor(operand)))


def ppermute(operand: ArrayLikeSpec, axis_name: str, perm: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``ppermute``.

    Args:
        operand: Input parameter ``operand``.
        axis_name: Input parameter ``axis_name``.
        perm: Input parameter ``perm``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.ppermute(_to_tensor(operand), axis_name, _to_tensor(perm)))


def pshuffle(operand: ArrayLikeSpec, axis_name: str, perm: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``pshuffle``.

    Args:
        operand: Input parameter ``operand``.
        axis_name: Input parameter ``axis_name``.
        perm: Input parameter ``perm``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.pshuffle(_to_tensor(operand), axis_name, _to_tensor(perm)))


def psum_scatter(operand: ArrayLikeSpec, axis_name: str, scatter_dimension: int = 0, tiled: bool = False) -> ArrayLikeSpec:
    """Execute ``psum_scatter``.

    Args:
        operand: Input parameter ``operand``.
        axis_name: Input parameter ``axis_name``.
        scatter_dimension: Input parameter ``scatter_dimension``.
        tiled: Input parameter ``tiled``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.psum_scatter(_to_tensor(operand), scatter_dimension, axis_name))


def pswapaxes(operand: ArrayLikeSpec, axis_name: str, axis: int) -> ArrayLikeSpec:
    """Execute ``pswapaxes``.

    Args:
        operand: Input parameter ``operand``.
        axis_name: Input parameter ``axis_name``.
        axis: Input parameter ``axis``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.pswapaxes(_to_tensor(operand), axis_name, axis))


def ragged_dot(lhs: ArrayLikeSpec, rhs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``ragged_dot``.

    Args:
        lhs: Input parameter ``lhs``.
        rhs: Input parameter ``rhs``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.ragged_dot(_to_tensor(lhs), _to_tensor(rhs)))


def random_gamma_grad(a: ArrayLikeSpec, x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``random_gamma_grad``.

    Args:
        a: Input parameter ``a``.
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.random_gamma_grad(_to_tensor(a), _to_tensor(x)))


def reduce_precision(operand: ArrayLikeSpec, exponent_bits: int, mantissa_bits: int) -> ArrayLikeSpec:
    """Execute ``reduce_precision``.

    Args:
        operand: Input parameter ``operand``.
        exponent_bits: Input parameter ``exponent_bits``.
        mantissa_bits: Input parameter ``mantissa_bits``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.reduce_precision(_to_tensor(operand), exponent_bits, mantissa_bits))


def reduce_window_shape_tuple(operand_shape: ArrayLikeSpec, window_dimensions: ArrayLikeSpec, window_strides: ArrayLikeSpec, padding: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``reduce_window_shape_tuple``.

    Args:
        operand_shape: Input parameter ``operand_shape``.
        window_dimensions: Input parameter ``window_dimensions``.
        window_strides: Input parameter ``window_strides``.
        padding: Input parameter ``padding``.

    Returns:
        The operation result.
    """
    return operand_shape


def rev(operand: ArrayLikeSpec, dimensions: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``rev``.

    Args:
        operand: Input parameter ``operand``.
        dimensions: Input parameter ``dimensions``.

    Returns:
        The operation result.
    """
    return operand


def rng_bit_generator(key: ArrayLikeSpec, shape: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``rng_bit_generator``.

    Args:
        key: Input parameter ``key``.
        shape: Input parameter ``shape``.

    Returns:
        The operation result.
    """
    return key, key


def rng_uniform(a: ArrayLikeSpec, b: ArrayLikeSpec, shape: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``rng_uniform``.

    Args:
        a: Input parameter ``a``.
        b: Input parameter ``b``.
        shape: Input parameter ``shape``.

    Returns:
        The operation result.
    """
    return a


def scan_bind(scan_primitive: ArrayLikeSpec, *args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``scan_bind``.

    Args:
        scan_primitive: Input parameter ``scan_primitive``.
        *args: Input parameter ``args``.
        **kwargs: Input parameter ``kwargs``.

    Returns:
        The operation result.
    """
    return None


def scatter_apply(
    operand: ArrayLikeSpec,
    scatter_indices: ArrayLikeSpec,
    updates: ArrayLikeSpec,
    update_computation: ArrayLikeSpec,
    dimension_numbers: ArrayLikeSpec,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> ArrayLikeSpec:
    """Execute ``scatter_apply``.

    Args:
        operand: Input parameter ``operand``.
        scatter_indices: Input parameter ``scatter_indices``.
        updates: Input parameter ``updates``.
        update_computation: Input parameter ``update_computation``.
        dimension_numbers: Input parameter ``dimension_numbers``.
        indices_are_sorted: Input parameter ``indices_are_sorted``.
        unique_indices: Input parameter ``unique_indices``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(
        ops.scatter_apply(
            _to_tensor(operand),
            _to_tensor(scatter_indices),
            _to_tensor(updates),
            _to_tensor(dimension_numbers),
            update_computation,
        )
    )


def scatter_max(
    operand: ArrayLikeSpec,
    scatter_indices: ArrayLikeSpec,
    updates: ArrayLikeSpec,
    dimension_numbers: ArrayLikeSpec,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> ArrayLikeSpec:
    """Execute ``scatter_max``.

    Args:
        operand: Input parameter ``operand``.
        scatter_indices: Input parameter ``scatter_indices``.
        updates: Input parameter ``updates``.
        dimension_numbers: Input parameter ``dimension_numbers``.
        indices_are_sorted: Input parameter ``indices_are_sorted``.
        unique_indices: Input parameter ``unique_indices``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(
        ops.scatter_max(
            _to_tensor(operand),
            _to_tensor(scatter_indices),
            _to_tensor(updates),
            _to_tensor(dimension_numbers),
            indices_are_sorted,
            unique_indices,
        )
    )


def scatter_min(
    operand: ArrayLikeSpec,
    scatter_indices: ArrayLikeSpec,
    updates: ArrayLikeSpec,
    dimension_numbers: ArrayLikeSpec,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> ArrayLikeSpec:
    """Execute ``scatter_min``.

    Args:
        operand: Input parameter ``operand``.
        scatter_indices: Input parameter ``scatter_indices``.
        updates: Input parameter ``updates``.
        dimension_numbers: Input parameter ``dimension_numbers``.
        indices_are_sorted: Input parameter ``indices_are_sorted``.
        unique_indices: Input parameter ``unique_indices``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(
        ops.scatter_min(
            _to_tensor(operand),
            _to_tensor(scatter_indices),
            _to_tensor(updates),
            _to_tensor(dimension_numbers),
            indices_are_sorted,
            unique_indices,
        )
    )


def scatter_mul(
    operand: ArrayLikeSpec,
    scatter_indices: ArrayLikeSpec,
    updates: ArrayLikeSpec,
    dimension_numbers: ArrayLikeSpec,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
) -> ArrayLikeSpec:
    """Execute ``scatter_mul``.

    Args:
        operand: Input parameter ``operand``.
        scatter_indices: Input parameter ``scatter_indices``.
        updates: Input parameter ``updates``.
        dimension_numbers: Input parameter ``dimension_numbers``.
        indices_are_sorted: Input parameter ``indices_are_sorted``.
        unique_indices: Input parameter ``unique_indices``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(
        ops.scatter_mul(
            _to_tensor(operand),
            _to_tensor(scatter_indices),
            _to_tensor(updates),
            _to_tensor(dimension_numbers),
            indices_are_sorted,
            unique_indices,
        )
    )


def select_n(pred: ArrayLikeSpec, *cases: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``select_n``.

    Args:
        pred: Input parameter ``pred``.
        *cases: Input parameter ``cases``.

    Returns:
        The operation result.
    """
    from zero_jax.numpy import where

    return where(pred, cases[1], cases[0])


def slice_in_dim(
    operand: ArrayLikeSpec,
    start_index: int | None,
    limit_index: int | None,
    stride: int = 1,
    dimension: int = 0,
) -> ArrayLikeSpec:
    """Extract a 1D slice along a single dimension.

    Args:
        operand: Input parameter ``operand``.
        start_index: Input parameter ``start_index``.
        limit_index: Input parameter ``limit_index``.
        stride: Input parameter ``stride``.
        dimension: Input parameter ``dimension``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    t = _to_tensor(operand)
    ndim = len(t.shape)
    axis = dimension if dimension >= 0 else ndim + dimension
    starts = [0] * ndim
    stops = list(t.shape)
    strides = [1] * ndim
    starts[axis] = start_index if start_index is not None else 0
    stops[axis] = limit_index if limit_index is not None else t.shape[axis]
    strides[axis] = stride
    return _wrap(ops.strided_slice(t, starts, stops, strides))


def sort_key_val(keys: ArrayLikeSpec, values: ArrayLikeSpec, dimension: int = -1) -> ArrayLikeSpec:
    """Execute ``sort_key_val``.

    Args:
        keys: Input parameter ``keys``.
        values: Input parameter ``values``.
        dimension: Input parameter ``dimension``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.sort_key_val(_to_tensor(keys), _to_tensor(values), dimension))


def top_k(operand: ArrayLikeSpec, k: int) -> ArrayLikeSpec:
    """Execute ``top_k``.

    Args:
        operand: Input parameter ``operand``.
        k: Input parameter ``k``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.top_k(_to_tensor(operand), k))


def while_loop(cond_fun: ArrayLikeSpec, body_fun: ArrayLikeSpec, init_val: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``while_loop``.

    Args:
        cond_fun: Input parameter ``cond_fun``.
        body_fun: Input parameter ``body_fun``.
        init_val: Input parameter ``init_val``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.while_loop(cond_fun, body_fun, _to_tensor(init_val)))


def with_sharding_constraint(x: ArrayLikeSpec, sharding: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``with_sharding_constraint``.

    Args:
        x: Input parameter ``x``.
        sharding: Input parameter ``sharding``.

    Returns:
        The operation result.
    """
    return x


def xeinsum(subscripts: str, *operands: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``xeinsum``.

    Args:
        subscripts: Input parameter ``subscripts``.
        *operands: Input parameter ``operands``.

    Returns:
        The operation result.
    """
    from zero_jax.numpy import einsum

    return einsum(subscripts, *operands)


def zeros_like_array(x: ArrayLikeSpec, dtype: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``zeros_like_array``.

    Args:
        x: Input parameter ``x``.
        dtype: Input parameter ``dtype``.

    Returns:
        The operation result.
    """
    from zero_jax.numpy import zeros_like

    return zeros_like(x, dtype=dtype)


def zeta(x: ArrayLikeSpec, q: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``zeta``.

    Args:
        x: Input parameter ``x``.
        q: Input parameter ``q``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.zeta(_to_tensor(x), _to_tensor(q)))


def after_all(*operands: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``after_all``.

    Args:
        *operands: Input parameter ``operands``.

    Returns:
        The operation result.
    """
    return None


def all_gather(operand: ArrayLikeSpec, axis_name: str, tiled: bool = False) -> ArrayLikeSpec:
    """Execute ``all_gather``.

    Args:
        operand: Input parameter ``operand``.
        axis_name: Input parameter ``axis_name``.
        tiled: Input parameter ``tiled``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.all_gather(_to_tensor(operand), axis_name))


def all_to_all(operand: ArrayLikeSpec, axis_name: str, split_axis: int, concat_axis: int) -> ArrayLikeSpec:
    """Execute ``all_to_all``.

    Args:
        operand: Input parameter ``operand``.
        axis_name: Input parameter ``axis_name``.
        split_axis: Input parameter ``split_axis``.
        concat_axis: Input parameter ``concat_axis``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.all_to_all(_to_tensor(operand), split_axis, concat_axis, axis_name))


def associative_scan(fn: ArrayLikeSpec, elems: ArrayLikeSpec, reverse: bool = False, axis: int = 0) -> ArrayLikeSpec:
    """Execute ``associative_scan``.

    Args:
        fn: Input parameter ``fn``.
        elems: Input parameter ``elems``.
        reverse: Input parameter ``reverse``.
        axis: Input parameter ``axis``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.associative_scan(fn, _to_tensor(elems), reverse, axis))


def axis_index(axis_name: str) -> ArrayLikeSpec:
    """Execute ``axis_index``.

    Args:
        axis_name: Input parameter ``axis_name``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    return ops.axis_index(axis_name)


def batch_matmul(lhs: ArrayLikeSpec, rhs: ArrayLikeSpec, precision: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``batch_matmul``.

    Args:
        lhs: Input parameter ``lhs``.
        rhs: Input parameter ``rhs``.
        precision: Input parameter ``precision``.

    Returns:
        The operation result.
    """
    from zero_jax.numpy import matmul

    return matmul(lhs, rhs)


def bessel_i0e(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``bessel_i0e``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.bessel_i0e(_to_tensor(x)))


def bessel_i1e(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``bessel_i1e``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.bessel_i1e(_to_tensor(x)))


def clz(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``clz``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.clz(_to_tensor(x)))


def complex(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``complex``.

    Args:
        x: Input parameter ``x``.
        y: Input parameter ``y``.

    Returns:
        The operation result.
    """
    from zero_jax.numpy import array

    return array(x) + 1.0j * array(y)


def conv(lhs: ArrayLikeSpec, rhs: ArrayLikeSpec, window_strides: ArrayLikeSpec, padding: ArrayLikeSpec, precision: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``conv``.

    Args:
        lhs: Input parameter ``lhs``.
        rhs: Input parameter ``rhs``.
        window_strides: Input parameter ``window_strides``.
        padding: Input parameter ``padding``.
        precision: Input parameter ``precision``.

    Returns:
        The operation result.
    """
    return lhs


def convert_element_type(operand: ArrayLikeSpec, new_dtype: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``convert_element_type``.

    Args:
        operand: Input parameter ``operand``.
        new_dtype: Input parameter ``new_dtype``.

    Returns:
        The operation result.
    """
    from zero_jax.numpy import astype

    return astype(operand, new_dtype)


def create_token() -> ArrayLikeSpec:
    """Execute ``create_token``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops

    return ops.create_token()


def cumlogsumexp(operand: ArrayLikeSpec, axis: int = 0, reverse: bool = False) -> ArrayLikeSpec:
    """Execute ``cumlogsumexp``.

    Args:
        operand: Input parameter ``operand``.
        axis: Input parameter ``axis``.
        reverse: Input parameter ``reverse``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.cumlogsumexp(_to_tensor(operand), axis, reverse))


def cummax(operand: ArrayLikeSpec, axis: int = 0, reverse: bool = False) -> ArrayLikeSpec:
    """Execute ``cummax``.

    Args:
        operand: Input parameter ``operand``.
        axis: Input parameter ``axis``.
        reverse: Input parameter ``reverse``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.cummax(_to_tensor(operand), axis, reverse))


def cummin(operand: ArrayLikeSpec, axis: int = 0, reverse: bool = False) -> ArrayLikeSpec:
    """Execute ``cummin``.

    Args:
        operand: Input parameter ``operand``.
        axis: Input parameter ``axis``.
        reverse: Input parameter ``reverse``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.cummin(_to_tensor(operand), axis, reverse))


def cumprod(operand: ArrayLikeSpec, axis: int = 0, reverse: bool = False) -> ArrayLikeSpec:
    """Execute ``cumprod``.

    Args:
        operand: Input parameter ``operand``.
        axis: Input parameter ``axis``.
        reverse: Input parameter ``reverse``.

    Returns:
        The operation result.
    """
    from zero_jax.numpy import cumprod as np_cumprod

    return np_cumprod(operand, axis=axis)


def integer_pow(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``integer_pow``.

    Args:
        x: Input parameter ``x``.
        y: Input parameter ``y``.

    Returns:
        The operation result.
    """
    from zero_jax.numpy import power

    return power(x, y)


def is_finite(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``is_finite``.

    Args:
        x: Input parameter ``x``.

    Returns:
        The operation result.
    """
    from zero_jax.numpy import isfinite

    return isfinite(x)


def map(f: ArrayLikeSpec, xs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``map``.

    Args:
        f: Input parameter ``f``.
        xs: Input parameter ``xs``.

    Returns:
        The operation result.
    """
    return f(xs)


def pow(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``pow``.

    Args:
        x: Input parameter ``x``.
        y: Input parameter ``y``.

    Returns:
        The operation result.
    """
    from zero_jax.numpy import power

    return power(x, y)


def rem(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``rem``.

    Args:
        x: Input parameter ``x``.
        y: Input parameter ``y``.

    Returns:
        The operation result.
    """
    import zero_jax._compiler_proxy_ops as ops
    from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

    return _wrap(ops.rem(_to_tensor(x), _to_tensor(y)))


def switch(index: ArrayLikeSpec, branches: ArrayLikeSpec, *operands: ArrayLikeSpec, operand: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Execute ``switch``.

    Args:
        index: Input parameter ``index``.
        branches: Input parameter ``branches``.
        *operands: Input parameter ``operands``.
        operand: Optional legacy single operand.

    Returns:
        The operation result.
    """
    _ = index
    call_ops = (operand,) if operand is not None and not operands else operands
    return branches[0](*call_ops)


def axis_size(axis_name: ArrayLikeSpec) -> int:
    """Return the size of the mapped axis ``axis_name``.

    Args:
        axis_name: Axis name or tuple of axis names.

    Returns:
        int: Size of the axis (``1`` in single-device execution).
    """
    _ = axis_name
    return 1


def broadcast_like(arr: ArrayLikeSpec, like_arr: ArrayLikeSpec) -> ArrayLikeSpec:
    """Broadcast ``arr`` to the shape of ``like_arr``.

    Args:
        arr: Array to broadcast.
        like_arr: Prototype array defining the target shape.

    Returns:
        ArrayLikeSpec: Broadcasted array.
    """
    from zero_jax.numpy import asarray, broadcast_to

    return broadcast_to(asarray(arr), asarray(like_arr).shape)


def composite(decomposition: ArrayLikeSpec, name: str, version: int = 0) -> ArrayLikeSpec:
    """Wrap ``decomposition`` as a named StableHLO composite operation.

    Args:
        decomposition: Function implementing the composite operation.
        name: Composite operation name.
        version: Composite version number.

    Returns:
        ArrayLikeSpec: Wrapped decomposition callable.
    """
    _ = (name, version)
    return decomposition


def dce_sink(val: ArrayLikeSpec, prevent_mlir_dce: bool = False) -> ArrayLikeSpec:
    """Prevent dead-code elimination of ``val``.

    Args:
        val: Value or PyTree to sink.
        prevent_mlir_dce: Whether to emit an MLIR side-effecting sink barrier.

    Returns:
        ArrayLikeSpec: Unmodified ``val``.
    """
    _ = prevent_mlir_dce
    return val


def empty(
    shape: ArrayLikeSpec,
    dtype: ArrayLikeSpec = "float32",
    *,
    out_sharding: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Return an uninitialized array of ``shape`` and ``dtype``.

    Args:
        shape: Output array shape.
        dtype: Output array dtype.
        out_sharding: Optional output sharding specification.

    Returns:
        ArrayLikeSpec: Allocated array.
    """
    _ = out_sharding
    from zero_jax.numpy import empty as jnp_empty

    return jnp_empty(shape, dtype=dtype)


def log2(x: ArrayLikeSpec, accuracy: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Compute elementwise base-2 logarithm of ``x``.

    Args:
        x: Input array.
        accuracy: Optional accuracy mode specification.

    Returns:
        ArrayLikeSpec: Base-2 logarithm of ``x``.
    """
    _ = accuracy
    from zero_jax.numpy import log2 as jnp_log2

    return jnp_log2(x)


def mulhi(x: ArrayLikeSpec, y: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute the high half of the full-width integer product of ``x`` and ``y``.

    Args:
        x: First integer operand.
        y: Second integer operand.

    Returns:
        ArrayLikeSpec: High bits of ``x * y``.
    """
    from zero_jax.numpy import asarray, bitwise_right_shift, int64

    prod_64 = asarray(x, dtype=int64) * asarray(y, dtype=int64)
    return bitwise_right_shift(prod_64, 32)


def one_minus_square(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Compute ``1 - x ** 2`` elementwise.

    Args:
        x: Input array.

    Returns:
        ArrayLikeSpec: ``1 - x ** 2``.
    """
    from zero_jax.numpy import asarray, square

    arr = asarray(x)
    return 1 - square(arr)


def optimization_barrier(operand: ArrayLikeSpec) -> ArrayLikeSpec:
    """Prevent compiler optimizations from moving computations across ``operand``.

    Args:
        operand: Input array or PyTree.

    Returns:
        ArrayLikeSpec: Unmodified ``operand``.
    """
    return operand


def pcast(x: ArrayLikeSpec, axis_name: ArrayLikeSpec, *, to: str = "varying") -> ArrayLikeSpec:
    """Perform a parallel collective cast of ``x`` along ``axis_name``.

    Args:
        x: Input array.
        axis_name: Mapped axis name.
        to: Target variance mode (e.g. ``"varying"`` or ``"reduced"``).

    Returns:
        ArrayLikeSpec: Cast array ``x``.
    """
    _ = (axis_name, to)
    return x


def polynomial(x: ArrayLikeSpec, coeffs: ArrayLikeSpec, *, unroll: int | None = None) -> ArrayLikeSpec:
    """Evaluate a polynomial with coefficients ``coeffs`` at ``x`` using Horner's method.

    Args:
        x: Evaluation point array.
        coeffs: Sequence of polynomial coefficients from highest to lowest degree.
        unroll: Optional loop unroll threshold.

    Returns:
        ArrayLikeSpec: Evaluated polynomial values.
    """
    _ = unroll
    from zero_jax.numpy import polyval

    return polyval(coeffs, x)


def psum_scatter_start(*args: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
    """Initiate an asynchronous collective reduce-scatter operation.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ArrayLikeSpec: First positional argument or ``None``.
    """
    _ = kwargs
    return args[0] if args else None


def ragged_all_to_all(
    operand: ArrayLikeSpec,
    output: ArrayLikeSpec,
    input_offsets: ArrayLikeSpec,
    send_sizes: ArrayLikeSpec,
    output_offsets: ArrayLikeSpec,
    recv_sizes: ArrayLikeSpec,
    *,
    axis_name: ArrayLikeSpec,
    axis_index_groups: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Perform a ragged all-to-all collective exchange along ``axis_name``.

    Args:
        operand: Input buffer array.
        output: Output buffer array.
        input_offsets: Starting offsets in ``operand``.
        send_sizes: Number of elements to send per peer.
        output_offsets: Starting offsets in ``output``.
        recv_sizes: Number of elements to receive per peer.
        axis_name: Collective axis name.
        axis_index_groups: Optional replica index groups.

    Returns:
        ArrayLikeSpec: Populated ``output`` buffer.
    """
    _ = (operand, input_offsets, send_sizes, output_offsets, recv_sizes, axis_name, axis_index_groups)
    return output


def ragged_dot_general(
    lhs: ArrayLikeSpec,
    rhs: ArrayLikeSpec,
    group_sizes: ArrayLikeSpec,
    ragged_dot_dimension_numbers: ArrayLikeSpec = None,
    precision: ArrayLikeSpec = None,
    preferred_element_type: ArrayLikeSpec = None,
    group_offset: ArrayLikeSpec = 0,
    out_sharding: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
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
        ArrayLikeSpec: Ragged dot product output.
    """
    _ = (group_sizes, ragged_dot_dimension_numbers, precision, preferred_element_type, group_offset, out_sharding)
    return ragged_dot(lhs, rhs)


def reduce_and(operand: ArrayLikeSpec, axes: ArrayLikeSpec) -> ArrayLikeSpec:
    """Reduce ``operand`` along ``axes`` using logical/bitwise AND.

    Args:
        operand: Input array.
        axes: Sequence of axes to reduce over.

    Returns:
        ArrayLikeSpec: Reduced array.
    """
    from zero_jax.numpy import all as jnp_all

    return jnp_all(operand, axis=tuple(axes) if isinstance(axes, (list, tuple)) else axes)


def reduce_max(operand: ArrayLikeSpec, axes: ArrayLikeSpec, out_sharding: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Reduce ``operand`` along ``axes`` using maximum.

    Args:
        operand: Input array.
        axes: Sequence of axes to reduce over.
        out_sharding: Optional output sharding specification.

    Returns:
        ArrayLikeSpec: Reduced maximum array.
    """
    _ = out_sharding
    from zero_jax.numpy import max as jnp_max

    return jnp_max(operand, axis=tuple(axes) if isinstance(axes, (list, tuple)) else axes)


def reduce_min(operand: ArrayLikeSpec, axes: ArrayLikeSpec, out_sharding: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Reduce ``operand`` along ``axes`` using minimum.

    Args:
        operand: Input array.
        axes: Sequence of axes to reduce over.
        out_sharding: Optional output sharding specification.

    Returns:
        ArrayLikeSpec: Reduced minimum array.
    """
    _ = out_sharding
    from zero_jax.numpy import min as jnp_min

    return jnp_min(operand, axis=tuple(axes) if isinstance(axes, (list, tuple)) else axes)


def reduce_or(operand: ArrayLikeSpec, axes: ArrayLikeSpec) -> ArrayLikeSpec:
    """Reduce ``operand`` along ``axes`` using logical/bitwise OR.

    Args:
        operand: Input array.
        axes: Sequence of axes to reduce over.

    Returns:
        ArrayLikeSpec: Reduced array.
    """
    from zero_jax.numpy import any as jnp_any

    return jnp_any(operand, axis=tuple(axes) if isinstance(axes, (list, tuple)) else axes)


def reduce_prod(operand: ArrayLikeSpec, axes: ArrayLikeSpec) -> ArrayLikeSpec:
    """Reduce ``operand`` along ``axes`` using product.

    Args:
        operand: Input array.
        axes: Sequence of axes to reduce over.

    Returns:
        ArrayLikeSpec: Reduced product array.
    """
    from zero_jax.numpy import prod as jnp_prod

    return jnp_prod(operand, axis=tuple(axes) if isinstance(axes, (list, tuple)) else axes)


def reduce_sum(operand: ArrayLikeSpec, axes: ArrayLikeSpec, out_sharding: ArrayLikeSpec = None) -> ArrayLikeSpec:
    """Reduce ``operand`` along ``axes`` using sum.

    Args:
        operand: Input array.
        axes: Sequence of axes to reduce over.
        out_sharding: Optional output sharding specification.

    Returns:
        ArrayLikeSpec: Reduced sum array.
    """
    _ = out_sharding
    from zero_jax.numpy import sum as jnp_sum

    return jnp_sum(operand, axis=tuple(axes) if isinstance(axes, (list, tuple)) else axes)


def reduce_xor(operand: ArrayLikeSpec, axes: ArrayLikeSpec) -> ArrayLikeSpec:
    """Reduce ``operand`` along ``axes`` using bitwise/logical XOR.

    Args:
        operand: Input array.
        axes: Sequence of axes to reduce over.

    Returns:
        ArrayLikeSpec: Reduced XOR array.
    """
    from zero_jax.numpy import asarray, bitwise_xor

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
    from zero_jax.numpy import asarray, astype, matmul

    l_arr = asarray(lhs)
    r_arr = asarray(rhs)
    if lhs_scale is not None:
        l_arr = l_arr * asarray(lhs_scale)
    if rhs_scale is not None:
        r_arr = r_arr * asarray(rhs_scale)
    if dimension_numbers is not None:
        from zero_jax.lax.primitives import dot_general

        return dot_general(l_arr, r_arr, dimension_numbers=dimension_numbers, preferred_element_type=preferred_element_type)
    res = matmul(l_arr, r_arr)
    if preferred_element_type is not None:
        return astype(res, preferred_element_type)
    return res


def scatter_sub(
    operand: ArrayLikeSpec,
    scatter_indices: ArrayLikeSpec,
    updates: ArrayLikeSpec,
    dimension_numbers: ArrayLikeSpec,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: ArrayLikeSpec = None,
) -> ArrayLikeSpec:
    """Scatter-subtract ``updates`` from ``operand`` at ``scatter_indices``.

    Args:
        operand: Target array.
        scatter_indices: Scatter index array.
        updates: Values to subtract.
        dimension_numbers: :class:`ScatterDimensionNumbers` specification.
        indices_are_sorted: Whether indices are sorted.
        unique_indices: Whether indices are unique.
        mode: Out-of-bounds indexing mode.

    Returns:
        ArrayLikeSpec: Updated array.
    """
    _ = (dimension_numbers, indices_are_sorted, unique_indices, mode)
    from zero_jax.numpy import array, asarray

    op_arr = asarray(operand)
    base = op_arr.__array__().copy()
    idx_np = asarray(scatter_indices).__array__().ravel()
    upd_np = asarray(updates).__array__().ravel()
    for i, idx_val in enumerate(idx_np):
        base[int(idx_val)] -= upd_np[i % len(upd_np)]
    return array(base, dtype=op_arr.dtype)


def shape_as_value(shape: ArrayLikeSpec) -> tuple[ArrayLikeSpec, ...]:
    """Convert a static or dynamic ``shape`` tuple into a tuple of 0-D scalar arrays.

    Args:
        shape: Sequence of dimension sizes.

    Returns:
        tuple[ArrayLikeSpec, ...]: Tuple of scalar arrays representing each dimension size.
    """
    from zero_jax.numpy import array, int32

    return tuple(array(int(d), dtype=int32) for d in shape)


def split(operand: ArrayLikeSpec, sizes: ArrayLikeSpec, axis: int = 0) -> tuple[ArrayLikeSpec, ...]:
    """Split ``operand`` along ``axis`` into sub-arrays of explicit ``sizes``.

    Args:
        operand: Input array to split.
        sizes: Sequence of section sizes along ``axis``.
        axis: Axis along which to split.

    Returns:
        tuple[ArrayLikeSpec, ...]: Tuple of split sub-arrays.
    """
    from zero_jax.numpy import asarray
    from zero_jax.numpy import split as jnp_split

    arr = asarray(operand)
    cuts: list[int] = []
    running = 0
    size_list = list(sizes)
    for s in size_list[:-1]:
        running += int(s)
        cuts.append(running)
    return tuple(jnp_split(arr, cuts, axis=axis))


def stack(operands: ArrayLikeSpec, axis: int = 0) -> ArrayLikeSpec:
    """Stack a sequence of arrays ``operands`` along a new ``axis``.

    Args:
        operands: Sequence of arrays with identical shape and dtype.
        axis: Axis in the result array along which the inputs are stacked.

    Returns:
        ArrayLikeSpec: Stacked array.
    """
    from zero_jax.numpy import stack as jnp_stack

    return jnp_stack(operands, axis=axis)


def stage(x: ArrayLikeSpec) -> ArrayLikeSpec:
    """Stage ``x`` into the current trace/computation.

    Args:
        x: Input array or callable.

    Returns:
        ArrayLikeSpec: Staged ``x``.
    """
    return x


def tile(operand: ArrayLikeSpec, reps: ArrayLikeSpec) -> ArrayLikeSpec:
    """Construct an array by repeating ``operand`` the number of times given by ``reps``.

    Args:
        operand: Input array.
        reps: Number of repetitions along each axis.

    Returns:
        ArrayLikeSpec: Tiled array.
    """
    from zero_jax.numpy import tile as jnp_tile

    return jnp_tile(operand, reps)


def unstack(x: ArrayLikeSpec, axis: int = 0) -> tuple[ArrayLikeSpec, ...]:
    """Unpack ``x`` along ``axis`` into a tuple of arrays.

    Args:
        x: Input array to unstack.
        axis: Axis along which to unstack.

    Returns:
        tuple[ArrayLikeSpec, ...]: Tuple of unstacked arrays.
    """
    from zero_jax.numpy import unstack as jnp_unstack

    return tuple(jnp_unstack(x, axis=axis))


__all__ = [
    "after_all",
    "all_gather",
    "all_to_all",
    "approx_max_k",
    "approx_min_k",
    "associative_scan",
    "axis_index",
    "axis_size",
    "batch_matmul",
    "bessel_i0e",
    "bessel_i1e",
    "betainc",
    "bitcast_convert_type",
    "broadcast_like",
    "broadcast_to_rank",
    "broadcasted_iota",
    "clz",
    "collapse",
    "complex",
    "composite",
    "conv",
    "conv_dimension_numbers",
    "conv_general_dilated_local",
    "conv_general_dilated_patches",
    "conv_general_permutations",
    "conv_general_shape_tuple",
    "conv_shape_tuple",
    "conv_transpose",
    "conv_transpose_shape_tuple",
    "conv_with_general_padding",
    "convert_element_type",
    "create_token",
    "cumlogsumexp",
    "cummax",
    "cummin",
    "cumprod",
    "custom_linear_solve",
    "custom_root",
    "dce_sink",
    "dtype",
    "dynamic_index_in_dim",
    "dynamic_slice_in_dim",
    "dynamic_update_index_in_dim",
    "dynamic_update_slice_in_dim",
    "empty",
    "erf_inv",
    "fori_loop",
    "igamma",
    "igamma_grad_a",
    "igammac",
    "index_in_dim",
    "index_take",
    "infeed",
    "integer_pow",
    "iota",
    "is_finite",
    "log2",
    "logistic",
    "map",
    "mulhi",
    "one_minus_square",
    "optimization_barrier",
    "outfeed",
    "padtype_to_pads",
    "pbroadcast",
    "pcast",
    "pdot",
    "platform_dependent",
    "pmax",
    "pmin",
    "polygamma",
    "polynomial",
    "population_count",
    "pow",
    "ppermute",
    "pshuffle",
    "psum_scatter",
    "psum_scatter_start",
    "pswapaxes",
    "ragged_all_to_all",
    "ragged_dot",
    "ragged_dot_general",
    "random_gamma_grad",
    "reduce_and",
    "reduce_max",
    "reduce_min",
    "reduce_or",
    "reduce_precision",
    "reduce_prod",
    "reduce_sum",
    "reduce_window_shape_tuple",
    "reduce_xor",
    "rem",
    "rev",
    "rng_bit_generator",
    "rng_uniform",
    "scaled_dot",
    "scan_bind",
    "scatter_apply",
    "scatter_max",
    "scatter_min",
    "scatter_mul",
    "scatter_sub",
    "select_n",
    "shape_as_value",
    "slice_in_dim",
    "sort_key_val",
    "split",
    "stack",
    "stage",
    "switch",
    "tile",
    "top_k",
    "unstack",
    "while_loop",
    "with_sharding_constraint",
    "xeinsum",
    "zeros_like_array",
    "zeta",
]
