"""Proxy module for ml_switcheroo_compiler.ops.creation."""

from typing import Any

import ml_switcheroo_compiler.ops.creation as _original_creation
from ml_switcheroo_compiler.ops.base import get_op

_MAPPINGS = {
    "zeros": "Zeros",
    "ones": "Ones",
    "full": "Full",
    "zeros_like": "ZerosLike",
    "ones_like": "OnesLike",
    "full_like": "FullLike",
    "arange": "Arange",
}


def __getattr__(name: str) -> Any:
    op_name = _MAPPINGS.get(name)
    if op_name:
        try:
            return get_op(op_name)()
        except KeyError:
            pass

    try:
        return getattr(_original_creation, name)
    except AttributeError:  # pragma: no cover
        pass  # pragma: no cover

    op_name = "".join(word.title() for word in name.split("_"))  # pragma: no cover
    try:  # pragma: no cover
        return get_op(op_name)()  # pragma: no cover
    except KeyError:  # pragma: no cover
        raise AttributeError(  # pragma: no cover
            f"module 'ml_switcheroo_compiler.ops.creation' has no attribute '{name}'"
        ) from None
