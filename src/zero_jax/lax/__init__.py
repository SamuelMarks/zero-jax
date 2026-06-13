"""LAX module for core primitive operations and control flow."""

from __future__ import annotations

from typing import Any
import ml_switcheroo_compiler

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
    gather,
    scatter,
    scatter_add,
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
    "gather",
    "scatter",
    "scatter_add",
    "reduce",
    "select",
    "clamp",
]
