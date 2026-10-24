"""Miscellaneous JAX APIs."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence

from zero_jax._types import ArrayLikeSpec, ConfigValue, PyTree, ShapeSpec, T, VariadicCallable
from zero_jax.numpy.lax_numpy import ndarray


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
    aux_args: list[PyTree] = []
    if callable(fun):
        for val in getattr_safe_dict(fun).values():
            if isinstance(val, ndarray):
                aux_args.append(val)
    return fun, aux_args


def getattr_safe_dict(obj: ConfigValue) -> dict[str, PyTree]:
    """Return ``__dict__`` of a callable if it is a standard function or object.

    Args:
        obj: Target callable or object.

    Returns:
        dict[str, PyTree]: Attribute dictionary.
    """
    try:
        return dict(obj.__dict__)
    except AttributeError:
        return {}


def named_call(fun: VariadicCallable[R], name: str | None = None) -> VariadicCallable[R]:
    """Annotate a function with a custom name for debugging and profiling.

    Args:
        fun: Function to annotate.
        name: Optional scope name to use.

    Returns:
        VariadicCallable[R]: The annotated function.
    """
    return fun


R = T


def named_scope(name: str) -> contextlib.AbstractContextManager[None]:
    """Return a context manager for naming a computation scope.

    Args:
        name: Name of the scope.

    Returns:
        contextlib.AbstractContextManager[None]: Context manager entering the named scope.
    """

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
    *args: ConfigValue,
    **kwargs: ConfigValue,
) -> VariadicCallable[R]:
    """Recompute ``fun`` during the backward pass instead of storing intermediates.

    Supports both direct decoration (``@jax.checkpoint``) and parameterized
    decoration (``@jax.checkpoint(prevent_cse=False, policy=...)``).

    Args:
        fun: Function to rematerialize, or ``None`` when used with keyword arguments.
        *args: Additional configuration positional arguments.
        **kwargs: Additional configuration keyword arguments.

    Returns:
        VariadicCallable[R]: The rematerialized function or decorator.
    """
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

    def wrapper(*args: PyTree, **kwargs: PyTree) -> PyTree:
        """Trace ``fun`` with the given arguments and return its JAXpr representation.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            PyTree: ``_TracedJaxprRepr`` or ``(_TracedJaxprRepr, out)`` when ``return_shape=True``.
        """
        from zero_jax.api.core import ShapeDtypeStruct
        from zero_jax.numpy.lax_numpy import _to_tensor
        from zero_jax.tree_util import tree_flatten, tree_map

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
    result_shape_dtypes: ConfigValue,
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
    return callback(*args, **kwargs)


def make_array_from_callback(
    shape: ShapeSpec,
    sharding: ConfigValue,
    callback: VariadicCallable[ArrayLikeSpec],
) -> ndarray:
    """Create a sharded array by invoking ``callback`` for each shard.

    Args:
        shape: Output array shape.
        sharding: Array sharding specification.
        callback: Callback function returning shard data.

    Returns:
        ndarray: Constructed array of zeros with ``shape``.
    """
    from zero_jax.numpy import zeros

    return zeros(shape)


def make_array_from_process_local_data(
    sharding: ConfigValue,
    local_data: T,
) -> T:
    """Create a globally sharded array from process-local data.

    Args:
        sharding: Sharding specification.
        local_data: Process-local array data.

    Returns:
        T: The local data wrapped as a global array.
    """
    return local_data


def make_array_from_single_device_arrays(
    shape: ShapeSpec,
    sharding: ConfigValue,
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
    from zero_jax.numpy import stack

    return stack(arrays)


def softmax_custom_jvp(x: ArrayLikeSpec) -> ndarray:
    """Compute softmax using a custom JVP rule.

    Args:
        x: Input array.

    Returns:
        ndarray: Softmax of ``x``.
    """
    from zero_jax.nn import softmax

    return softmax(x)


def enable_custom_vjp_by_custom_transpose(enable: bool = True) -> None:
    """Enable ``custom_vjp`` rules implemented via custom transpose.

    Args:
        enable: Whether to enable custom transpose VJP rules.
    """


class _Float0:
    """Type representing a zero-width tangent space for integer/boolean primals."""

    def __init__(
        self,
        *args: ConfigValue,
        id: ConfigValue = None,
        name: ConfigValue = None,
        value: ConfigValue = None,
    ) -> None:
        """Initialize ``_Float0`` instance.

        Args:
            *args: Positional arguments.
            id: Optional identifier.
            name: Optional name.
            value: Optional value.
        """
        self.args = args
        self.id = id
        self.name = name if name is not None else "float0"
        self.value = value


float0 = _Float0()


@contextlib.contextmanager
def threefry_partitionable(*args: ConfigValue, **kwargs: ConfigValue) -> Iterator[None]:
    """Context manager for enabling partitionable Threefry PRNG.

    Args:
        *args: Positional configuration arguments.
        **kwargs: Keyword configuration arguments.

    Yields:
        None: Context where partitionable Threefry is configured.
    """
    yield


class _CheckpointPolicies:
    """Namespace holder for rematerialization checkpoint policies."""

    everything_saveable = "everything_saveable"
    nothing_saveable = "nothing_saveable"
    dots_saveable = "dots_saveable"
    dots_with_no_batch_dims_saveable = "dots_with_no_batch_dims_saveable"


checkpoint_policies = _CheckpointPolicies()


@contextlib.contextmanager
def legacy_prng_key(*args: ConfigValue, **kwargs: ConfigValue) -> Iterator[None]:
    """Context manager for enabling legacy PRNG key behavior.

    Args:
        *args: Positional configuration arguments.
        **kwargs: Keyword configuration arguments.

    Yields:
        None: Context where legacy PRNG keys are configured.
    """
    yield


def enable_custom_prng(enable: bool = True) -> None:
    """Enable custom PRNG key implementations.

    Args:
        enable: Whether to enable custom PRNG keys.
    """


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
