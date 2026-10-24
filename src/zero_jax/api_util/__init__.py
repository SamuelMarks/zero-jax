"""Frontend API routing for jax.api_util."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec, ArrayProtocol, ConfigValue
from zero_jax.api.transformations import ShapedArray
from zero_jax.dtypes import result_type
from zero_jax.tree_util.pytree import PyTreeDef, tree_flatten, tree_unflatten


def argnums_partial(
    f: ArrayLikeSpec = None,
    dyn_argnums: ArrayLikeSpec = 0,
    args: ArrayLikeSpec = (),
    require_static_args_hashable: bool = True,
) -> ArrayLikeSpec:
    """Partially apply ``f`` to static arguments while leaving ``dyn_argnums`` dynamic.

    Args:
        f: Function to partially apply.
        dyn_argnums: Integer or sequence of dynamic argument indices.
        args: Full tuple of arguments.
        require_static_args_hashable: Whether static arguments must be hashable.

    Returns:
        ArrayLikeSpec: Tuple of ``(partially_applied_fun, dynamic_args)``.
    """
    if f is None or not callable(f):
        return (None, ())
    dyn_set = {dyn_argnums} if isinstance(dyn_argnums, int) else set(dyn_argnums)
    full_args = tuple(args) if isinstance(args, (tuple, list)) else (args,)
    dyn_vals = tuple(a for idx, a in enumerate(full_args) if idx in dyn_set)

    def _wrapped(*new_dyn: ArrayLikeSpec, **kwargs: ArrayLikeSpec) -> ArrayLikeSpec:
        """Invoke ``f`` with merged static and dynamic arguments.

        Args:
            *new_dyn: Dynamic argument values.
            **kwargs: Keyword arguments.

        Returns:
            ArrayLikeSpec: Result of ``f``.
        """
        merged: list[ArrayLikeSpec] = []
        dyn_iter = iter(new_dyn)
        for idx, orig in enumerate(full_args):
            if idx in dyn_set:
                merged.append(next(dyn_iter, orig))
            else:
                merged.append(orig)
        return f(*merged, **kwargs)

    return _wrapped, dyn_vals


def donation_vector(
    donate_argnums: ArrayLikeSpec = (),
    donate_argnames: ArrayLikeSpec = (),
    in_tree: PyTreeDef | ConfigValue = None,
) -> tuple[bool, ...]:
    """Return a tuple of boolean buffer-donation flags for each flattened leaf.

    Args:
        donate_argnums: Donated positional argument indices.
        donate_argnames: Donated keyword argument names.
        in_tree: Optional input ``PyTreeDef``.

    Returns:
        tuple[bool, ...]: Tuple of boolean donation flags.
    """
    donated_nums = {donate_argnums} if isinstance(donate_argnums, int) else set(donate_argnums or ())
    if isinstance(in_tree, PyTreeDef):
        result: list[bool] = []
        for idx, child_def in enumerate(in_tree.children_defs):
            is_donated = idx in donated_nums
            result.extend([is_donated] * max(1, child_def.num_leaves))
        return tuple(result)
    return tuple(True for _ in donated_nums)


def flatten_axes(
    name: str = "axes",
    treedef: PyTreeDef | ConfigValue = None,
    axis_tree: ArrayLikeSpec = 0,
    *,
    kws: bool = False,
) -> list[ArrayLikeSpec]:
    """Broadcast and flatten an axis specification ``axis_tree`` to match ``treedef`` leaves.

    Args:
        name: Parameter name used in error messages.
        treedef: Target ``PyTreeDef`` structure.
        axis_tree: Axis specification PyTree or prefix.
        kws: Whether ``treedef`` includes a keyword argument dictionary.

    Returns:
        list[ArrayLikeSpec]: Flat list of axis specs, one per leaf of ``treedef``.
    """
    if isinstance(treedef, PyTreeDef):
        if isinstance(axis_tree, int) or axis_tree is None:
            return [axis_tree] * treedef.num_leaves
        flat_axes, _ = tree_flatten(axis_tree)
        if len(flat_axes) == 1 and treedef.num_leaves > 1:
            return flat_axes * treedef.num_leaves
        return flat_axes
    flat_axes, _ = tree_flatten(axis_tree)
    return flat_axes


def flatten_fun(
    fun: ArrayLikeSpec = None,
    in_tree: PyTreeDef | ConfigValue = None,
) -> tuple[ArrayLikeSpec, ArrayLikeSpec]:
    """Transform a function taking ``(args, kwargs)`` PyTrees into one taking flat leaves.

    Args:
        fun: Callable accepting ``(*args, **kwargs)``.
        in_tree: ``PyTreeDef`` of ``(args, kwargs)``.

    Returns:
        tuple[ArrayLikeSpec, ArrayLikeSpec]: ``(flat_fun, out_tree_thunk)``.
    """
    holder: list[PyTreeDef | None] = [None]

    def _flat_fun(*flat_args: ArrayLikeSpec) -> list[ArrayLikeSpec]:
        """Execute ``fun`` on unflattened arguments and return flat output leaves.

        Args:
            *flat_args: Flattened leaf arguments.

        Returns:
            list[ArrayLikeSpec]: Flattened output leaves.
        """
        if not callable(fun):
            return []
        if isinstance(in_tree, PyTreeDef):
            unflat = tree_unflatten(in_tree, list(flat_args))
            if isinstance(unflat, tuple) and len(unflat) == 2 and isinstance(unflat[1], dict):
                pos_args, kw_args = unflat
                res = fun(*pos_args, **kw_args)
            elif isinstance(unflat, (tuple, list)):
                res = fun(*unflat)
            else:
                res = fun(unflat)
        else:
            res = fun(*flat_args)
        out_leaves, out_def = tree_flatten(res)
        holder[0] = out_def
        return out_leaves

    return _flat_fun, lambda: holder[0]


def flatten_fun_nokwargs(
    fun: ArrayLikeSpec = None,
    in_tree: PyTreeDef | ConfigValue = None,
) -> tuple[ArrayLikeSpec, ArrayLikeSpec]:
    """Transform a function taking positional PyTree args into one taking flat leaves.

    Args:
        fun: Callable accepting positional arguments.
        in_tree: ``PyTreeDef`` of positional arguments.

    Returns:
        tuple[ArrayLikeSpec, ArrayLikeSpec]: ``(flat_fun, out_tree_thunk)``.
    """
    return flatten_fun(fun, in_tree)


def rebase_donate_argnums(
    donate_argnums: ArrayLikeSpec = (),
    static_argnums: ArrayLikeSpec = (),
) -> tuple[int, ...]:
    """Shift ``donate_argnums`` indices down to account for removed ``static_argnums``.

    Args:
        donate_argnums: Integer or sequence of donated argument indices.
        static_argnums: Integer or sequence of static argument indices.

    Returns:
        tuple[int, ...]: Rebased donated argument indices in sorted order.
    """
    donated = {donate_argnums} if isinstance(donate_argnums, int) else set(donate_argnums or ())
    static = {static_argnums} if isinstance(static_argnums, int) else set(static_argnums or ())
    rebased: list[int] = []
    for d in sorted(donated):
        if d not in static:
            shift = sum(1 for s in static if s < d)
            rebased.append(d - shift)
    return tuple(rebased)


def safe_map(f: ArrayLikeSpec = None, *args: ArrayLikeSpec) -> list[ArrayLikeSpec]:
    """Map callable ``f`` over one or more equal-length sequences.

    Args:
        f: Function to map across elements.
        *args: Sequences of equal length.

    Returns:
        list[ArrayLikeSpec]: List of mapped values.

    Raises:
        ValueError: If input sequences have mismatched lengths.
    """
    if not callable(f) or not args:
        return []
    seqs = [list(a) for a in args]
    first_len = len(seqs[0])
    for s in seqs[1:]:
        if len(s) != first_len:
            raise ValueError(f"safe_map requires equal lengths, got {first_len} and {len(s)}")
    return [f(*items) for items in zip(*seqs)]


def shaped_abstractify(x: ArrayLikeSpec = 0.0) -> ShapedArray:
    """Convert a concrete array or Python scalar into a :class:`ShapedArray` abstract value.

    Args:
        x: Concrete array or scalar value.

    Returns:
        ShapedArray: Abstract shape and dtype representation.
    """
    if isinstance(x, ShapedArray):
        return x
    if isinstance(x, ArrayProtocol):
        return ShapedArray(x.shape, x.dtype)
    return ShapedArray((), result_type(x))


__all__ = [
    "argnums_partial",
    "donation_vector",
    "flatten_axes",
    "flatten_fun",
    "flatten_fun_nokwargs",
    "rebase_donate_argnums",
    "safe_map",
    "shaped_abstractify",
]
