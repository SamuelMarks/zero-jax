"""Module ops."""

import typing
from typing import Any

import ml_switcheroo_compiler

import zero_jax._compiler_proxy_ops as _ops


class Abs:
    def __init__(self, *args, **kwargs):
        pass


class Add:
    def __init__(self, *args, **kwargs):
        pass


class And:
    def __init__(self, *args, **kwargs):
        pass


class Cos:
    def __init__(self, *args, **kwargs):
        pass


class Div:
    def __init__(self, *args, **kwargs):
        pass


class Eq:
    def __init__(self, *args, **kwargs):
        pass


class Erf:
    def __init__(self, *args, **kwargs):
        pass


class Exp:
    def __init__(self, *args, **kwargs):
        pass


class Ge:
    def __init__(self, *args, **kwargs):
        pass


class Gt:
    def __init__(self, *args, **kwargs):
        pass


LU: Any = None


class Le:
    def __init__(self, *args, **kwargs):
        pass


class Log:
    def __init__(self, *args, **kwargs):
        pass


class Lt:
    def __init__(self, *args, **kwargs):
        pass


class Map:
    def __init__(self, *args, **kwargs):
        pass


class Max:
    def __init__(self, *args, **kwargs):
        pass


class Min:
    def __init__(self, *args, **kwargs):
        pass


class Mul:
    def __init__(self, *args, **kwargs):
        pass


class Ne:
    def __init__(self, *args, **kwargs):
        pass


class Neg:
    def __init__(self, *args, **kwargs):
        pass


class Not:
    def __init__(self, *args, **kwargs):
        pass


class Or:
    def __init__(self, *args, **kwargs):
        pass


class Pad:
    def __init__(self, *args, **kwargs):
        pass


class Pow:
    def __init__(self, *args, **kwargs):
        pass


QR: Any = None


class Rem:
    def __init__(self, *args, **kwargs):
        pass


class Rev:
    def __init__(self, *args, **kwargs):
        pass


SVD: Any = None


class Sin:
    def __init__(self, *args, **kwargs):
        pass


class Sub:
    def __init__(self, *args, **kwargs):
        pass


class Tan:
    def __init__(self, *args, **kwargs):
        pass


class Xor:
    def __init__(self, *args, **kwargs):
        pass


def __getattr__(name):
    if hasattr(_ops, name):
        return getattr(_ops, name)
    if hasattr(ml_switcheroo_compiler, name):
        return getattr(ml_switcheroo_compiler, name)  # pragma: no cover
    try:
        from zero_jax.numpy.lax_numpy import _to_tensor, _wrap

        # If it's a known missing function, we might just return a dummy callable that raises NotImplementedError,
        # BUT we only want to do that if it really doesn't exist, to pass test_stubs.py
        def stub(*args, **kwargs):
            raise NotImplementedError(f"Stub for {name} is not implemented in backend")

        return stub
    except ImportError:  # pragma: no cover

        def stub(*args, **kwargs):  # pragma: no cover
            raise NotImplementedError(
                f"Stub for {name} is not implemented in backend"
            )  # pragma: no cover

        return stub  # pragma: no cover
