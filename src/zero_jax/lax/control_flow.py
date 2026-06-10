"""Control flow primitives for zero_jax."""

from typing import Callable, Any
from ml_switcheroo.tracing import _tracer
from ml_switcheroo.core.config import config
import ml_switcheroo.control_flow as cf
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap


def cond(pred: Any, true_fn: Callable, false_fn: Callable, *operands: Any) -> Any:
    def wrapped_true():
        return _to_tensor(true_fn(*operands))

    def wrapped_false():
        return _to_tensor(false_fn(*operands))

    return _wrap(cf.cond(_to_tensor(pred), wrapped_true, wrapped_false))


def scan(f: Callable, init: Any, xs: Any, length: int = None) -> Any:
    if xs is None:
        if length is None:
            raise ValueError("length must be provided if xs is None")
        xs = [0] * length
    elif config.eager_mode:
        pass  # xs is iterable

    if config.eager_mode and not _tracer.is_tracing:
        carry = init
        ys = []
        for x in xs:
            carry, y = f(carry, x)
            ys.append(y)
        return carry, ys
    else:
        # tracing
        carry, ys = cf.scan(f, _to_tensor(init), _to_tensor(xs))
        return _wrap(carry), _wrap(ys)


def stop_gradient(x: Any) -> Any:
    # Actually switcheroo doesn't have stop_gradient yet, just pass through
    return x
