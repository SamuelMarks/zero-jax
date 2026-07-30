"""Patches for ml-switcheroo-compiler missing eager registrations."""

from typing import Any

# Force import so we can overwrite
import ml_switcheroo_compiler.backends.numpy.eager.linalg_extras
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
