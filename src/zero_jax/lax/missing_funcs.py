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


def switch(index: ArrayLikeSpec, branches: ArrayLikeSpec, *operands: ArrayLikeSpec) -> ArrayLikeSpec:
    """Execute ``switch``.

    Args:
        index: Input parameter ``index``.
        branches: Input parameter ``branches``.
        *operands: Input parameter ``operands``.

    Returns:
        The operation result.
    """
    return branches[0](*operands)


__all__ = [
    "after_all",
    "all_gather",
    "all_to_all",
    "approx_max_k",
    "approx_min_k",
    "associative_scan",
    "axis_index",
    "batch_matmul",
    "bessel_i0e",
    "bessel_i1e",
    "betainc",
    "bitcast_convert_type",
    "broadcast_to_rank",
    "broadcasted_iota",
    "clz",
    "collapse",
    "complex",
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
    "dtype",
    "dynamic_index_in_dim",
    "dynamic_slice_in_dim",
    "dynamic_update_index_in_dim",
    "dynamic_update_slice_in_dim",
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
    "logistic",
    "map",
    "outfeed",
    "padtype_to_pads",
    "pbroadcast",
    "pdot",
    "platform_dependent",
    "pmax",
    "pmin",
    "polygamma",
    "population_count",
    "pow",
    "ppermute",
    "pshuffle",
    "psum_scatter",
    "pswapaxes",
    "ragged_dot",
    "random_gamma_grad",
    "reduce_precision",
    "reduce_window_shape_tuple",
    "rem",
    "rev",
    "rng_bit_generator",
    "rng_uniform",
    "scan_bind",
    "scatter_apply",
    "scatter_max",
    "scatter_min",
    "scatter_mul",
    "select_n",
    "slice_in_dim",
    "sort_key_val",
    "switch",
    "top_k",
    "while_loop",
    "with_sharding_constraint",
    "xeinsum",
    "zeros_like_array",
    "zeta",
]
