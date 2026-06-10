"""Module docstring."""

from typing import Any
import ml_switcheroo

"Module docstring."
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
