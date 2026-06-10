"""schedules."""


def linear_schedule(
    init_value: float,
    end_value: float,
    transition_steps: int,
    transition_begin: int = 0,
):
    """linear_schedule."""

    def schedule(count):
        """schedule."""

        import numpy as np
        import zero_jax.numpy as jnp

        # count can be array
        c = np.clip(count - transition_begin, 0, transition_steps)
        frac = c / transition_steps
        return (1 - frac) * init_value + frac * end_value

    return schedule
