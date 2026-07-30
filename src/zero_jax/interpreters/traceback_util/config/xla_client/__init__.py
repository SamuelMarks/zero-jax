"""Module xla_client."""

import typing
from typing import Any

import ml_switcheroo_compiler

import zero_jax._compiler_proxy_ops as _ops


class ArrayImpl:
    def __init__(self, *args, **kwargs):
        pass


class Client:
    def __init__(self, *args, **kwargs):
        pass


class DeviceList:
    def __init__(self, *args, **kwargs):
        pass


class FftType:
    def __init__(self, *args, **kwargs):
        pass


class Frame:
    def __init__(self, *args, **kwargs):
        pass


class HloSharding:
    def __init__(self, *args, **kwargs):
        pass


class Layout:
    def __init__(self, *args, **kwargs):
        pass


class Mapping:
    def __init__(self, *args, **kwargs):
        pass


class Memory:
    def __init__(self, *args, **kwargs):
        pass


class OpMetadata:
    def __init__(self, *args, **kwargs):
        pass


class OpSharding:
    def __init__(self, *args, **kwargs):
        pass


class PaddingType:
    def __init__(self, *args, **kwargs):
        pass


class PjRtLayout:
    def __init__(self, *args, **kwargs):
        pass


class PmapSharding:
    def __init__(self, *args, **kwargs):
        pass


class ProgramShape:
    def __init__(self, *args, **kwargs):
        pass


class Protocol:
    def __init__(self, *args, **kwargs):
        pass


class ReplicaGroup:
    def __init__(self, *args, **kwargs):
        pass


class Sequence:
    def __init__(self, *args, **kwargs):
        pass


class Shape:
    def __init__(self, *args, **kwargs):
        pass


class ShapeIndex:
    def __init__(self, *args, **kwargs):
        pass


class Sharding:
    def __init__(self, *args, **kwargs):
        pass


class Traceback:
    def __init__(self, *args, **kwargs):
        pass


class XlaBuilder:
    def __init__(self, *args, **kwargs):
        pass


class XlaOp:
    def __init__(self, *args, **kwargs):
        pass


def bfloat16(*args: Any, **kwargs: Any) -> Any:
    pass


def logger(*args: Any, **kwargs: Any) -> Any:
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
