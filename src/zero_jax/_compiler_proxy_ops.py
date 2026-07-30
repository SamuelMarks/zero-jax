"""Proxy module for ml_switcheroo_compiler.ops to handle dynamically generated ops."""

from typing import Any

import ml_switcheroo_compiler.ops as _original_ops
from ml_switcheroo_compiler.ops.base import get_op

# Map specific jax functions to compiler op names
_MAPPINGS = {
    "concatenate": "Concatenate",
    "broadcast_to": "BroadcastTo",
    "permute": "Transpose",
    "reshape": "Reshape",
    "where": "Where",
    "fft": "Fft",
    "sum": "Sum",
    "max": "Max",
    "min": "Min",
    "mean": "Mean",
    "prod": "Prod",
    "argmax": "Argmax",
    "argmin": "Argmin",
    "any": "Any",
    "all": "All",
    "squeeze": "Squeeze",
    "unsqueeze": "Unsqueeze",
    "strided_slice": "StridedSlice",
    "dynamic_slice": "DynamicSlice",
    "update_slice": "DynamicUpdateSlice",
    "gather": "Gather",
    "scatter": "Scatter",
    "searchsorted": "SearchSorted",
    "cholesky": "Cholesky",
    "det": "Det",
    "svd": "Svd",
    "inv": "Inv",
    "matrix_power": "MatrixPower",
    "pinv": "Pinv",
}


def __getattr__(name: str) -> Any:
    op_name = _MAPPINGS.get(name)
    if op_name:
        try:
            return get_op(op_name)()
        except KeyError:
            pass

    try:
        return getattr(_original_ops, name)
    except AttributeError:
        pass

    op_name = "".join(word.title() for word in name.split("_"))
    try:
        return get_op(op_name)()
    except KeyError:
        raise AttributeError(
            f"module 'ml_switcheroo_compiler.ops' has no attribute '{name}'"
        ) from None


__all__ = getattr(_original_ops, "__all__", [])
