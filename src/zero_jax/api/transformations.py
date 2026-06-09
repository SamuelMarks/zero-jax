"""Transformations for zero_jax."""

from typing import Callable
import functools


def jit(fun: Callable) -> Callable:
    # JIT compiler caching logic

    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        # We need a proper PyTree flattening and shape signature to cache
        # For this prototype, we just pass through
        return fun(*args, **kwargs)

    return wrapper


def grad(fun: Callable) -> Callable:
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        # Full tracing logic to trace 'fun', then call compiler_grad
        # This is a placeholder for the API
        return fun(*args, **kwargs)  # Should return gradient

    return wrapper


def value_and_grad(fun: Callable) -> Callable:
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        return fun(*args, **kwargs), fun(*args, **kwargs)

    return wrapper


def vmap(fun: Callable) -> Callable:
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        return fun(*args, **kwargs)

    return wrapper

import contextlib
@contextlib.contextmanager
def disable_jit(disable=True):
    yield

def pmap(fun, axis_name=None, in_axes=0, out_axes=0, static_broadcasted_argnums=(), devices=None, backend=None, axis_size=None, donate_argnums=(), global_arg_shapes=None):
    from .transformations import vmap
    return vmap(fun)
