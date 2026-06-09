"""Optax schedules.

This module implements various learning rate schedules.
"""

from typing import Any, Callable, Sequence, Union, Optional, Tuple, Iterable, Mapping, Literal, Type, Dict
import numpy as np
import collections.abc

class filterlib:
    class Filter: pass
class rnglib:
    class Rngs: pass
class variables:
    class Variable: pass
class chex:
    class Array: pass
    class Numeric: pass
    class Scalar: pass
class core:
    class Shape: pass
class optax:
    class _src:
        class base:
            class GradientTransformationExtraArgs: pass
class base:
    class GradientTransformation: pass
    class Schedule: pass
class jax:
    class Array: pass
    class Device: pass
    class _src:
        class typing:
            class SupportsDType: pass
M = Any
A = Any
UNSPECIFIED = None
_UNSPECIFIED = None
default_kernel_init = None
default_bias_init = None
default_embed_init = None
lax = Any
FrozenDict = Any
KeyArray = Any
RealNumeric = Any
LoRAParam = Any
dot_product_attention = None


# Type aliases
Array = Any
Numeric = Any
Schedule = Any


def constant_schedule(value: Numeric) -> Schedule:
    """Constructs a constant schedule.

    Args:
        value: Constant value.

    Returns:
        Schedule function.
    """

    def schedule(step: Numeric) -> Numeric:
        return value

    return schedule


def cosine_decay_schedule(
    init_value: float, decay_steps: int, alpha: float = 0.0, exponent: float = 1.0
) -> Schedule:
    """Returns a function which implements cosine learning rate decay.

    Args:
        init_value: Initial value.
        decay_steps: Decay steps.
        alpha: Alpha parameter.
        exponent: Exponent parameter.

    Returns:
        Schedule function.
    """

    def schedule(step: Numeric) -> Numeric:
        step = np.minimum(step, decay_steps)
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * step / decay_steps))
        decayed = (1.0 - alpha) * cosine_decay + alpha
        return init_value * (decayed**exponent)

    return schedule


def cosine_onecycle_schedule(
    transition_steps: int,
    peak_value: float,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 10000.0,
) -> Schedule:
    """Returns a function which implements the onecycle learning rate schedule.

    Args:
        transition_steps: Transition steps.
        peak_value: Peak value.
        pct_start: Percentage start.
        div_factor: Div factor.
        final_div_factor: Final div factor.

    Returns:
        Schedule function.
    """
    init_value = peak_value / div_factor
    final_value = init_value / final_div_factor
    warmup_steps = int(transition_steps * pct_start)
    decay_steps = transition_steps - warmup_steps

    def schedule(step: Numeric) -> Numeric:
        if step < warmup_steps:
            # Cosine warmup
            progress = step / warmup_steps
            cosine_warmup = 0.5 * (1.0 - np.cos(np.pi * progress))
            return init_value + (peak_value - init_value) * cosine_warmup
        else:
            # Cosine decay
            progress = (step - warmup_steps) / decay_steps
            progress = np.minimum(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            return final_value + (peak_value - final_value) * cosine_decay

    return schedule


def exponential_decay(
    init_value: float,
    transition_steps: int,
    decay_rate: float,
    transition_begin: int = 0,
    staircase: bool = False,
    end_value: Optional[float] = None,
) -> Schedule:
    """Constructs a schedule with either continuous or discrete exponential decay.

    Args:
        init_value: Initial value.
        transition_steps: Transition steps.
        decay_rate: Decay rate.
        transition_begin: Transition begin.
        staircase: Staircase.
        end_value: End value.

    Returns:
        Schedule function.
    """

    def schedule(step: Numeric) -> Numeric:
        p = step - transition_begin
        if staircase:
            p = np.floor(p / transition_steps)
        else:
            p = p / transition_steps

        decayed = init_value * (decay_rate**p)
        if end_value is not None:
            decayed = np.maximum(decayed, end_value)

        return np.where(step < transition_begin, init_value, decayed)

    return schedule


def inject_hyperparams(
    inner_factory: Callable[..., Any],
    static_args: Union[str, Iterable[str]] = (),
    hyperparam_dtype: Optional[Any] = None,
) -> Callable[..., Any]:
    """Wrapper to injects stateful hyperparameters into GradientTransformations.

    Args:
        inner_factory: Inner factory.
        static_args: Static args.
        hyperparam_dtype: Hyperparam dtype.

    Returns:
        A gradient transformation with injected hyperparams.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return inner_factory(*args, **kwargs)

    return wrapper


def inject_stateful_hyperparams(
    inner_factory: Callable[..., Any],
    static_args: Union[str, Iterable[str]] = (),
    hyperparam_dtype: Optional[Any] = None,
) -> Callable[..., Any]:
    """Wrapper to injects stateful hyperparameters into GradientTransformations.

    Args:
        inner_factory: Inner factory.
        static_args: Static args.
        hyperparam_dtype: Hyperparam dtype.

    Returns:
        A gradient transformation with injected stateful hyperparams.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return inner_factory(*args, **kwargs)

    return wrapper


def join_schedules(
    schedules: Sequence[Schedule], boundaries: Sequence[int]
) -> Schedule:
    """Sequentially apply multiple schedules.

    Args:
        schedules: Schedules.
        boundaries: Boundaries.

    Returns:
        Schedule function.
    """

    def schedule(step: Numeric) -> Numeric:
        step = np.asarray(step)
        # Find which schedule to use
        idx = np.sum(step >= np.array(boundaries))

        if idx == 0:
            return schedules[0](step)

        # We need to compute the step relative to the boundary
        # Actually JAX join_schedules does not reset step
        return schedules[idx](step)

    return schedule


def linear_onecycle_schedule(
    transition_steps: int,
    peak_value: float,
    pct_start: float = 0.3,
    pct_final: float = 0.85,
    div_factor: float = 25.0,
    final_div_factor: float = 10000.0,
) -> Schedule:
    """Returns a learning rate with three linear phases.

    Args:
        transition_steps: Transition steps.
        peak_value: Peak value.
        pct_start: Pct start.
        pct_final: Pct final.
        div_factor: Div factor.
        final_div_factor: Final div factor.

    Returns:
        Schedule function.
    """
    init_value = peak_value / div_factor
    final_value = init_value / final_div_factor

    warmup_steps = int(transition_steps * pct_start)
    decay_steps = int(transition_steps * pct_final) - warmup_steps
    final_steps = transition_steps - warmup_steps - decay_steps

    def schedule(step: Numeric) -> Numeric:
        if step < warmup_steps:
            return init_value + (peak_value - init_value) * (step / warmup_steps)
        elif step < warmup_steps + decay_steps:
            return peak_value - (peak_value - init_value) * (
                (step - warmup_steps) / decay_steps
            )
        else:
            progress = (step - warmup_steps - decay_steps) / max(1, final_steps)
            progress = min(1.0, progress)
            return init_value - (init_value - final_value) * progress

    return schedule


def linear_schedule(
    init_value: Numeric,
    end_value: Numeric,
    transition_steps: int,
    transition_begin: int = 0,
) -> Schedule:
    """Schedule with linear transition from ``init_value`` to ``end_value``.

    Args:
        init_value: Initial value.
        end_value: End value.
        transition_steps: Transition steps.
        transition_begin: Transition begin.

    Returns:
        Schedule function.
    """

    def schedule(step: Numeric) -> Numeric:
        p = (step - transition_begin) / transition_steps
        p = np.clip(p, 0.0, 1.0)
        return init_value + p * (end_value - init_value)

    return schedule


def piecewise_constant_schedule(
    init_value: float, boundaries_and_scales: Optional[Dict[int, float]] = None
) -> Schedule:
    """Returns a function which implements a piecewise constant schedule.

    Args:
        init_value: Initial value.
        boundaries_and_scales: Boundaries and scales.

    Returns:
        Schedule function.
    """
    if boundaries_and_scales is None:
        boundaries_and_scales = {}

    boundaries = sorted(list(boundaries_and_scales.keys()))
    scales = [boundaries_and_scales[b] for b in boundaries]

    def schedule(step: Numeric) -> Numeric:
        v = init_value
        for b, s in zip(boundaries, scales):
            if step >= b:
                v = init_value * s
        return v

    return schedule


def piecewise_interpolate_schedule(
    interpolate_type: str,
    init_value: float,
    boundaries_and_scales: Optional[Dict[int, float]] = None,
) -> Schedule:
    """Returns a function which implements a piecewise interpolated schedule.

    Args:
        interpolate_type: Interpolate type.
        init_value: Initial value.
        boundaries_and_scales: Boundaries and scales.

    Returns:
        Schedule function.
    """
    if boundaries_and_scales is None:
        boundaries_and_scales = {}

    boundaries = sorted(list(boundaries_and_scales.keys()))
    scales = [boundaries_and_scales[b] for b in boundaries]

    def schedule(step: Numeric) -> Numeric:
        step_val = float(np.asarray(step))
        if not boundaries or step_val <= boundaries[0]:
            return init_value

        idx = int(np.sum(step_val >= np.array(boundaries))) - 1

        start_step = boundaries[idx]
        start_val = init_value * scales[idx]

        if idx < len(boundaries) - 1:
            end_step = boundaries[idx + 1]
            end_val = init_value * scales[idx + 1]
        else:
            return start_val

        progress = (step_val - start_step) / (end_step - start_step)
        progress = np.clip(progress, 0.0, 1.0)

        if interpolate_type == "linear":
            return start_val + progress * (end_val - start_val)
        else:
            raise ValueError(f"Unsupported interpolate_type: {interpolate_type}")

    return schedule


def polynomial_schedule(
    init_value: Numeric,
    end_value: Numeric,
    power: Numeric,
    transition_steps: int,
    transition_begin: int = 0,
) -> Schedule:
    """Constructs a schedule with polynomial transition from init to end value.

    Args:
        init_value: Initial value.
        end_value: End value.
        power: Power.
        transition_steps: Transition steps.
        transition_begin: Transition begin.

    Returns:
        Schedule function.
    """

    def schedule(step: Numeric) -> Numeric:
        p = (step - transition_begin) / transition_steps
        p = np.clip(p, 0.0, 1.0)
        return (init_value - end_value) * ((1.0 - p) ** power) + end_value

    return schedule


def sgdr_schedule(cosine_kwargs: Optional[Dict[str, Any]] = None) -> Schedule:
    """SGD with warm restarts.

    Args:
        cosine_kwargs: Cosine kwargs sequence defining the cycles.

    Returns:
        Schedule function.
    """
    if cosine_kwargs is None:
        cosine_kwargs = [{"init_value": 0.1, "decay_steps": 10}]

    def schedule(step: Numeric) -> Numeric:
        curr_step = step
        for kwargs in cosine_kwargs:
            decay_steps = kwargs.get("decay_steps", 10)
            if curr_step < decay_steps:
                init_val = kwargs.get("init_value", 0.1)
                alpha = kwargs.get("alpha", 0.0)
                exponent = kwargs.get("exponent", 1.0)

                cosine_decay = 0.5 * (1.0 + np.cos(np.pi * curr_step / decay_steps))
                decayed = (1.0 - alpha) * cosine_decay + alpha
                return init_val * (decayed**exponent)
            curr_step -= decay_steps

        # If beyond all cycles, use the end of the last cycle
        last_kwargs = cosine_kwargs[-1]
        init_val = last_kwargs.get("init_value", 0.1)
        alpha = last_kwargs.get("alpha", 0.0)
        exponent = last_kwargs.get("exponent", 1.0)
        return init_val * (alpha**exponent)

    return schedule


def warmup_constant_schedule(
    init_value: float, peak_value: float, warmup_steps: int
) -> Schedule:
    """Linear warmup followed by constant schedule i.e no decay.

    Args:
        init_value: Initial value.
        peak_value: Peak value.
        warmup_steps: Warmup steps.

    Returns:
        Schedule function.
    """

    def schedule(step: Numeric) -> Numeric:
        p = step / max(1, warmup_steps)
        p = np.clip(p, 0.0, 1.0)
        return init_value + p * (peak_value - init_value)

    return schedule


def warmup_cosine_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float = 0.0,
    exponent: float = 1.0,
) -> Schedule:
    """Linear warmup followed by cosine decay.

    Args:
        init_value: Initial value.
        peak_value: Peak value.
        warmup_steps: Warmup steps.
        decay_steps: Decay steps.
        end_value: End value.
        exponent: Exponent.

    Returns:
        Schedule function.
    """

    def schedule(step: Numeric) -> Numeric:
        if step < warmup_steps:
            p = step / max(1, warmup_steps)
            return init_value + p * (peak_value - init_value)
        else:
            p = (step - warmup_steps) / max(1, decay_steps)
            p = np.clip(p, 0.0, 1.0)
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * p))
            return end_value + (peak_value - end_value) * (cosine_decay**exponent)

    return schedule


def warmup_exponential_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    transition_steps: int,
    decay_rate: float,
    transition_begin: int = 0,
    staircase: bool = False,
    end_value: Optional[float] = None,
) -> Schedule:
    """Linear warmup followed by exponential decay.

    Args:
        init_value: Initial value.
        peak_value: Peak value.
        warmup_steps: Warmup steps.
        transition_steps: Transition steps.
        decay_rate: Decay rate.
        transition_begin: Transition begin.
        staircase: Staircase.
        end_value: End value.

    Returns:
        Schedule function.
    """

    def schedule(step: Numeric) -> Numeric:
        if step < warmup_steps:
            p = step / max(1, warmup_steps)
            return init_value + p * (peak_value - init_value)
        else:
            p = step - warmup_steps - transition_begin
            if p < 0:
                return peak_value

            if staircase:
                p = np.floor(p / transition_steps)
            else:
                p = p / transition_steps

            decayed = peak_value * (decay_rate**p)
            if end_value is not None:
                decayed = np.maximum(decayed, end_value)
            return decayed

    return schedule
