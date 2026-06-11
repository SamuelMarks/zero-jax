"""Core API module for JAX-like transformations (jit, grad, vmap, etc.)."""

from typing import Any
import ml_switcheroo

from .transformations import (
    jit,
    grad,
    value_and_grad,
    vmap,
    disable_jit,
    pmap,
    eval_shape,
)

__all__ = ["jit", "grad", "value_and_grad", "vmap", "disable_jit", "pmap, eval_shape"]
