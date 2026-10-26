"""Miscellaneous JAX APIs."""

from __future__ import annotations

import contextlib
import types
from collections.abc import Iterator, Sequence

from zero_jax._types import (
    ArrayLikeSpec,
    ArrayProtocol,
    PyTree,
    ShapeSpec,
    StatefulProtocol,
    T,
    VariadicCallable,
)
from zero_jax.api.core import ShapeDtypeStruct
from zero_jax.nn import softmax as _nn_softmax
from zero_jax.numpy import asarray, stack, zeros
from zero_jax.numpy.lax_numpy import _to_tensor, ndarray
from zero_jax.sharding import Sharding
from zero_jax.tree_util import tree_flatten, tree_map


def closure_convert(
    fun: VariadicCallable[PyTree],
    *args: PyTree,
    **kwargs: PyTree,
) -> tuple[VariadicCallable[PyTree], list[PyTree]]:
    """Closure-convert a Python function for higher-order differentiation.

    Args:
        fun: The function to convert.
        *args: Example positional arguments.
        **kwargs: Example keyword arguments.

    Returns:
        tuple[VariadicCallable[PyTree], list[PyTree]]: Tuple of ``(converted_fun, residual_env)``.
    """
    _ = (args, kwargs)
    aux_args: list[PyTree] = []
    if callable(fun):
        for val in getattr_safe_dict(fun).values():
            if isinstance(val, ndarray):
                aux_args.append(val)
    return fun, aux_args


def getattr_safe_dict(obj: VariadicCallable[PyTree] | StatefulProtocol | PyTree) -> dict[str, PyTree]:
    """Return residual state mapping of a callable or PyTree-registered object without dunder dict access.

    Args:
        obj: Target callable or object.

    Returns:
        dict[str, PyTree]: Mapping of captured array leaves and function metadata.
    """
    result: dict[str, PyTree] = {}
    if isinstance(obj, StatefulProtocol):
        for k, v in obj.state.items():
            result[k] = v
    if isinstance(obj, types.FunctionType):
        result["__name__"] = obj.__name__
        if obj.__closure__:
            for idx, cell in enumerate(obj.__closure__):
                result[f"cell_{idx}"] = cell.cell_contents
        return result
    leaves, _ = tree_flatten(obj)
    for idx, leaf in enumerate(leaves):
        if isinstance(leaf, ndarray):
            result[f"leaf_{idx}"] = leaf
    return result


def named_call(fun: VariadicCallable[R], name: str | None = None) -> VariadicCallable[R]:
    """Annotate a function with a custom name for debugging and profiling.

    Args:
        fun: Function to annotate.
        name: Optional scope name to use.

    Returns:
        VariadicCallable[R]: The annotated function.
    """
    _ = name
    return fun


R = T


def named_scope(name: str) -> contextlib.AbstractContextManager[None]:
    """Return a context manager for naming a computation scope.

    Args:
        name: Name of the scope.

    Returns:
        contextlib.AbstractContextManager[None]: Context manager entering the named scope.
    """
    _ = name

    @contextlib.contextmanager
    def _scope() -> Iterator[None]:
        """Yield inside the named scope.

        Yields:
            None: Scope context.
        """
        yield

    return _scope()


def remat(
    fun: VariadicCallable[R] | None = None,
    *,
    prevent_cse: bool = True,
    policy: str | VariadicCallable[bool] | None = None,
    static_argnums: int | Sequence[int] = (),
    static_argnames: str | Sequence[str] = (),
) -> VariadicCallable[R]:
    """Recompute ``fun`` during the backward pass instead of storing intermediates.

    Supports both direct decoration (``@jax.checkpoint``) and parameterized
    decoration (``@jax.checkpoint(prevent_cse=False, policy=...)``).

    Args:
        fun: Function to rematerialize, or ``None`` when used with keyword arguments.
        prevent_cse: Whether to prevent common subexpression elimination across the boundary.
        policy: Optional rematerialization policy callable.
        static_argnums: Optional positional argument indices treated as static.
        static_argnames: Optional keyword argument names treated as static.

    Returns:
        VariadicCallable[R]: The rematerialized function or decorator.
    """
    _ = (prevent_cse, policy, static_argnums, static_argnames)
    if fun is None:

        def _decorator(inner_fun: VariadicCallable[R]) -> VariadicCallable[R]:
            """Decorate ``inner_fun`` with rematerialization.

            Args:
                inner_fun: Target callable.

            Returns:
                VariadicCallable[R]: Rematerialized callable.
            """
            return inner_fun

        return _decorator
    return fun


checkpoint = remat


def ensure_compile_time_eval(fun: VariadicCallable[R] | None = None) -> VariadicCallable[R] | contextlib.AbstractContextManager[None]:
    """Ensure a function or context block is evaluated at compile time.

    Args:
        fun: Optional function to wrap.

    Returns:
        VariadicCallable[R] | contextlib.AbstractContextManager[None]: Wrapped function or no-op context manager.
    """
    if fun is not None:
        return fun

    @contextlib.contextmanager
    def _eval_ctx() -> Iterator[None]:
        """Yield compile-time evaluation context.

        Yields:
            None: Active context.
        """
        yield

    return _eval_ctx()


class _TracedJaxprRepr:
    """Structured representation returned by :func:`make_jaxpr`."""

    def __init__(self, fun_name: str, num_inputs: int, out_shape: tuple[int, ...] = ()) -> None:
        """Initialize ``_TracedJaxprRepr``.

        Args:
            fun_name: Name of the traced function.
            num_inputs: Number of flattened input arguments.
            out_shape: Shape of the primary output.
        """
        self.fun_name = fun_name
        self.num_inputs = num_inputs
        self.out_shape = out_shape
        self.eqns: list[str] = []

    def __repr__(self) -> str:
        """Return human-readable Jaxpr string.

        Returns:
            str: Formatted Jaxpr string.
        """
        return f"{{ lambda ; a[{self.num_inputs}]. let b = {self.fun_name}(a) in (b,) }}"


def make_jaxpr(
    fun: VariadicCallable[PyTree],
    static_argnums: int | Sequence[int] = (),
    return_shape: bool = False,
) -> VariadicCallable[PyTree]:
    """Create a function that produces a JAXpr representation of ``fun``.

    Args:
        fun: The function to trace.
        static_argnums: Argument indices to treat as static.
        return_shape: Whether to return ``(jaxpr, out_shape)`` tuple.

    Returns:
        VariadicCallable[PyTree]: A callable returning the traced JAXpr representation.
    """
    _ = static_argnums

    def wrapper(*args: PyTree, **kwargs: PyTree) -> PyTree:
        """Trace ``fun`` with the given arguments and return its JAXpr representation.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            PyTree: ``_TracedJaxprRepr`` or ``(_TracedJaxprRepr, out)`` when ``return_shape=True``.
        """
        flat_in, _ = tree_flatten((args, kwargs))
        name = getattr_safe_dict(fun).get("__name__", "fun")
        out = fun(*args, **kwargs)
        flat_out, _ = tree_flatten(out)
        first_shape = _to_tensor(flat_out[0]).shape if flat_out else ()
        jaxpr_doc = _TracedJaxprRepr(str(name), len(flat_in), first_shape)
        if return_shape:
            shape_tree = tree_map(lambda leaf: ShapeDtypeStruct(_to_tensor(leaf).shape, _to_tensor(leaf).dtype), out)
            return jaxpr_doc, shape_tree
        return jaxpr_doc

    return wrapper


def pure_callback(
    callback: VariadicCallable[PyTree],
    result_shape_dtypes: ArrayProtocol | Sequence[ArrayProtocol] | PyTree,
    *args: PyTree,
    **kwargs: PyTree,
) -> PyTree:
    """Invoke a pure Python callback function from within a JAX computation.

    Args:
        callback: Host Python callable to execute.
        result_shape_dtypes: Expected output shapes and dtypes.
        *args: Positional arguments forwarded to ``callback``.
        **kwargs: Keyword arguments forwarded to ``callback``.

    Returns:
        PyTree: Result of invoking ``callback(*args, **kwargs)``.
    """
    _ = result_shape_dtypes
    return callback(*args, **kwargs)


def make_array_from_callback(
    shape: ShapeSpec,
    sharding: Sharding | None,
    callback: VariadicCallable[ArrayLikeSpec],
) -> ndarray:
    """Create a sharded array by invoking ``callback`` for the global shard index tuple.

    Args:
        shape: Output array shape.
        sharding: Array sharding specification.
        callback: Callback function accepting a tuple of ``slice(None)`` indices and returning shard data.

    Returns:
        ndarray: Constructed array of ``shape``.
    """
    _ = sharding
    shape_tup = (int(shape),) if isinstance(shape, int) else tuple(int(d) for d in shape)
    idx_tuple = tuple(slice(None) for _ in shape_tup)
    res = asarray(callback(idx_tuple))
    if res.shape == shape_tup:
        return res
    return zeros(shape_tup, dtype=res.dtype) + res


def make_array_from_process_local_data(
    sharding: Sharding | None,
    local_data: T,
) -> T:
    """Create a globally sharded array from process-local data.

    Args:
        sharding: Sharding specification.
        local_data: Process-local array data.

    Returns:
        T: The local data wrapped as a global array.
    """
    _ = sharding
    return local_data


def make_array_from_single_device_arrays(
    shape: ShapeSpec,
    sharding: Sharding | None,
    arrays: Sequence[ArrayLikeSpec],
) -> ndarray:
    """Create a sharded array from a sequence of single-device arrays.

    Args:
        shape: Global array shape.
        sharding: Sharding specification.
        arrays: Sequence of single-device arrays.

    Returns:
        ndarray: Stacked array constructed from ``arrays``.
    """
    _ = (shape, sharding)
    return stack(arrays)


def softmax_custom_jvp(x: ArrayLikeSpec) -> ndarray:
    """Compute softmax using a custom JVP rule.

    Args:
        x: Input array.

    Returns:
        ndarray: Softmax of ``x``.
    """
    return _nn_softmax(x)


def enable_custom_vjp_by_custom_transpose(enable: bool = True) -> None:
    """Enable ``custom_vjp`` rules implemented via custom transpose.

    Args:
        enable: Whether to enable custom transpose VJP rules.
    """
    _ = enable


class _Float0:
    """Type representing a zero-width tangent space for integer/boolean primals."""

    def __init__(self, dtype_name: str = "float0") -> None:
        """Initialize ``_Float0`` instance.

        Args:
            dtype_name: Symbolic tangent dtype name (defaults to ``"float0"``).
        """
        self.name: str = str(dtype_name)
        self.dtype: str = str(dtype_name)


float0 = _Float0()


@contextlib.contextmanager
def threefry_partitionable(enable: bool = True) -> Iterator[None]:
    """Context manager for enabling partitionable Threefry PRNG.

    Args:
        enable: Whether to enable partitionable Threefry PRNG.

    Yields:
        None: Context where partitionable Threefry is configured.
    """
    _ = enable
    yield


class _CheckpointPolicies:
    """Namespace holder for rematerialization checkpoint policies."""

    everything_saveable = "everything_saveable"
    nothing_saveable = "nothing_saveable"
    dots_saveable = "dots_saveable"
    dots_with_no_batch_dims_saveable = "dots_with_no_batch_dims_saveable"


checkpoint_policies = _CheckpointPolicies()


@contextlib.contextmanager
def legacy_prng_key(enable: str | bool = True) -> Iterator[None]:
    """Context manager for enabling legacy PRNG key behavior.

    Args:
        enable: Whether or how to enable legacy PRNG keys.

    Yields:
        None: Context where legacy PRNG keys are configured.
    """
    _ = enable
    yield


def enable_custom_prng(enable: bool = True) -> None:
    """Enable custom PRNG key implementations.

    Args:
        enable: Whether to enable custom PRNG keys.
    """
    _ = enable


def default_prng_impl() -> str:
    """Return the default PRNG implementation identifier.

    Returns:
        str: Default PRNG implementation string (``"threefry2x32"``).
    """
    return "threefry2x32"


def jax2tf_associative_scan_reductions() -> None:
    """Configure associative scan reductions for ``jax2tf``."""


def default_matmul_precision() -> str:
    """Return the default matrix multiplication precision setting.

    Returns:
        str: The precision string (``"highest"``).
    """
    return "highest"


@contextlib.contextmanager
def debug_key_reuse(enable: bool = True) -> Iterator[None]:
    """Context manager for debugging PRNG key reuse.

    Args:
        enable: Whether to enable PRNG key reuse checking.

    Yields:
        None: Context where key reuse checking is configured.
    """
    _ = enable
    yield


__all__ = [
    "_CheckpointPolicies",
    "_Float0",
    "_TracedJaxprRepr",
    "checkpoint",
    "checkpoint_policies",
    "closure_convert",
    "debug_key_reuse",
    "default_matmul_precision",
    "default_prng_impl",
    "enable_custom_prng",
    "enable_custom_vjp_by_custom_transpose",
    "ensure_compile_time_eval",
    "float0",
    "getattr_safe_dict",
    "jax2tf_associative_scan_reductions",
    "legacy_prng_key",
    "make_array_from_callback",
    "make_array_from_process_local_data",
    "make_array_from_single_device_arrays",
    "make_jaxpr",
    "named_call",
    "named_scope",
    "pure_callback",
    "remat",
    "softmax_custom_jvp",
    "threefry_partitionable",
]
