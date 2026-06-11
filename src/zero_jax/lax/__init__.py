"""LAX module for core primitive operations and control flow."""

from typing import Any
import ml_switcheroo

from .control_flow import cond, scan, stop_gradient
from .primitives import (
    add,
    sub,
    mul,
    div,
    broadcast,
    broadcast_in_dim,
    reshape,
    transpose,
    slice,
    dynamic_slice,
    dynamic_update_slice,
    reduce,
    select,
    clamp,
)

__all__ = [
    "cond",
    "scan",
    "stop_gradient",
    "add",
    "sub",
    "mul",
    "div",
    "broadcast",
    "broadcast_in_dim",
    "reshape",
    "transpose",
    "slice",
    "dynamic_slice",
    "dynamic_update_slice",
    "reduce",
    "select",
    "clamp",
]
