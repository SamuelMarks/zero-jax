"""Frontend API routing for jax.interpreters.xla.xb.hardware_utils."""

from typing import Any
from . import glob


def num_available_tpu_chips_and_device_id(*args: Any, **kwargs: Any) -> Any:
    """Returns the device id and number of TPU chips attached through PCI."""
    raise NotImplementedError(
        "num_available_tpu_chips_and_device_id not yet implemented in zero-jax"
    )


from . import os
from . import pathlib


def tpu_enhanced_barrier_supported(*args: Any, **kwargs: Any) -> Any:
    """Returns if tpu_enhanced_barrier flag is supported on this TPU version."""
    raise NotImplementedError(
        "tpu_enhanced_barrier_supported not yet implemented in zero-jax"
    )
