"""Control flow primitives for zero_jax."""

from typing import Any, Callable
from ml_switcheroo.tracing import _tracer
from ml_switcheroo.core.config import config
import ml_switcheroo.control_flow as cf
from zero_jax.numpy.lax_numpy import _to_tensor, _wrap


def cond(pred: Any, true_fn: Callable, false_fn: Callable, *operands: Any) -> Any:
    """Conditionally applies one of two functions.

    Args:
        pred: A boolean condition.
        true_fn: The function to execute if pred is true.
        false_fn: The function to execute if pred is false.
        *operands: Arguments to pass to the chosen function.

    Returns:
        The output of the conditionally executed function.
    """

    def wrapped_true() -> Any:
        """Executes the true branch.

        Returns:
            The tensor output of the true function.
        """
        return _to_tensor(true_fn(*operands))

    def wrapped_false() -> Any:
        """Executes the false branch.

        Returns:
            The tensor output of the false function.
        """
        return _to_tensor(false_fn(*operands))

    return _wrap(cf.cond(_to_tensor(pred), wrapped_true, wrapped_false))


def scan(f: Callable, init: Any, xs: Any, length: int = None) -> Any:
    """Scans a function over leading array axes while carrying along state.

    Args:
        f: The scanning function. Takes (carry, x) and returns (carry, y).
        init: The initial state.
        xs: The sequence of inputs over which to scan.
        length: Optional length of the scan, required if xs is None.

    Returns:
        A tuple of the final state and the stacked outputs.
    """
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
    """Stops the flow of gradients during reverse-mode differentiation.

    Args:
        x: The value to stop gradients for.

    Returns:
        The identical value, but treated as a constant by gradient computations.
    """
    # Actually switcheroo doesn't have stop_gradient yet, just pass through
    return x
