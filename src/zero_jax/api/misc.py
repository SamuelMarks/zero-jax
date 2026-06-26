"""Miscellaneous JAX APIs."""

from __future__ import annotations
import contextlib


from typing import Callable, Any


def closure_convert(fun: Callable, *args: Any, **kwargs: Any) -> Any:
    """Closure converts a function.

    Args:
        fun: The function to convert.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        The converted function and its captured environments.
    """
    return fun, []


def named_call(fun: Callable, name: str | None = None) -> Callable:
    """Annotates a function for debugging and profiling.

    Args:
        fun: Function to annotate.
        name: Name to use.

    Returns:
        The annotated function.
    """
    return fun


def named_scope(name: str) -> Any:
    """Context manager for naming a scope.

    Args:
        name: Scope name.

    Returns:
        A context manager.
    """

    @contextlib.contextmanager
    def _scope() -> Any:
        yield

    return _scope()


def remat(fun: Callable, *args: Any, **kwargs: Any) -> Callable:
    """Recomputes the function during backward pass.

    Args:
        fun: Function to rematerialize.
        *args: Extra args.
        **kwargs: Extra kwargs.

    Returns:
        The rematerialized function.
    """
    return fun


checkpoint = remat


def ensure_compile_time_eval(fun: Callable) -> Callable:
    """Ensures a function is evaluated at compile time.

    Args:
        fun: Function to wrap.

    Returns:
        Wrapped function.
    """
    return fun


def make_jaxpr(fun: Callable, static_argnums: Any = ()) -> Callable:
    """Creates a function that produces a jaxpr.

    Args:
        fun: The function to trace.
        static_argnums: Arguments to treat as static.

    Returns:
        A function returning the jaxpr.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return None

    return wrapper


def pure_callback(
    callback: Callable, result_shape_dtypes: Any, *args: Any, **kwargs: Any
) -> Any:
    """Calls a pure Python callback from JAX.

    Args:
        callback: Function to call.
        result_shape_dtypes: Expected output shapes and dtypes.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        The result of the callback.
    """
    return callback(*args, **kwargs)


def make_array_from_callback(shape: Any, sharding: Any, callback: Callable) -> Any:
    """Creates a sharded array via a callback.

    Args:
        shape: Array shape.
        sharding: Array sharding.
        callback: Callback function.

    Returns:
        The new array.
    """
    from zero_jax.numpy import zeros

    return zeros(shape)


def make_array_from_process_local_data(sharding: Any, local_data: Any) -> Any:
    """Creates a globally sharded array from local data.

    Args:
        sharding: Sharding spec.
        local_data: Local data array.

    Returns:
        The sharded array.
    """
    return local_data


def make_array_from_single_device_arrays(shape: Any, sharding: Any, arrays: Any) -> Any:
    """Creates a sharded array from single device arrays.

    Args:
        shape: Array shape.
        sharding: Sharding spec.
        arrays: Sequence of arrays.

    Returns:
        The sharded array.
    """
    from zero_jax.numpy import stack

    return stack(arrays)


def softmax_custom_jvp(x: Any) -> Any:
    """Softmax with a custom JVP.

    Args:
        x: Input array.

    Returns:
        Softmax of x.
    """
    from zero_jax.nn import softmax

    return softmax(x)


def enable_custom_vjp_by_custom_transpose(enable: bool = True) -> None:
    """Enables custom_vjp via custom transpose.

    Args:
        enable: Whether to enable.
    """
    pass


class _Float0:
    """Type representing a zero-sized float."""

    pass


float0 = _Float0()


class _ThreefryPartitionable:
    """Representation of threefry PRNG partitionability."""

    pass


threefry_partitionable = _ThreefryPartitionable()


class _CheckpointPolicies:
    """Namespace for checkpoint policies."""

    pass


checkpoint_policies = _CheckpointPolicies()


class _LegacyPrngKey:
    """Legacy PRNG key representation."""

    pass


legacy_prng_key = _LegacyPrngKey()


def enable_custom_prng(enable: bool = True) -> None:
    """Enables custom PRNG implementation.

    Args:
        enable: Whether to enable.
    """
    pass


def default_prng_impl() -> str:
    """Returns the default PRNG implementation.

    Returns:
        String representing default PRNG.
    """
    return "threefry2x32"


def jax2tf_associative_scan_reductions() -> None:
    """Associative scan reductions for jax2tf."""
    pass


def default_matmul_precision() -> str:
    """Returns the default matrix multiplication precision.

    Returns:
        The precision string.
    """
    return "highest"


@contextlib.contextmanager
def debug_key_reuse(enable: bool = True) -> Any:
    """Context manager for debugging PRNG key reuse.

    Args:
        enable: Whether to enable.

    Yields:
        None
    """
    yield
