"""PRNG state manipulation."""

from typing import Any
import ml_switcheroo.random as random
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap


def split(key: Any, num: int = 2) -> Any:
    """Splits a PRNG key into `num` new keys."""
    if key is None:
        return None
    return _wrap(random.split(_to_tensor(key), num))


def fold_in(key: Any, data: Any) -> Any:
    """Folds in data to a PRNG key."""
    if key is None:
        return None
    return _wrap(random.fold_in(_to_tensor(key), data))
