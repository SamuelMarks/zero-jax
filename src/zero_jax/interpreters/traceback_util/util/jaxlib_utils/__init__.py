"""jaxlib_utils stub."""

from __future__ import annotations

from zero_jax._types import ArrayLikeSpec, ConfigValue


class CallOp:
    """Frontend class for ``CallOp``."""


import typing

import ml_switcheroo_compiler

import zero_jax._compiler_proxy_ops as _ops


def safe_map(f: ArrayLikeSpec, *args: ArrayLikeSpec) -> ArrayLikeSpec:
    """Map a function over sequences, checking they have the same length.

    Args:
        f: Input parameter ``f``.
        *args: Input parameter ``args``.

    Returns:
        The operation result.
    """
    if not args:
        return []
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, "Lengths must match"
    return list(map(f, *args))


def safe_zip(*args: ArrayLikeSpec) -> ArrayLikeSpec:
    """Zip sequences, checking they have the same length.

    Args:
        *args: Input parameter ``args``.

    Returns:
        The operation result.
    """
    if not args:
        return []
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, "Lengths must match"
    return list(zip(*args))
