"""losses."""

import zero_jax.numpy as jnp


def l2_loss(predictions, targets):
    """l2_loss."""
    return jnp.multiply(0.5, jnp.power(jnp.subtract(predictions, targets), 2))


def softmax_cross_entropy(logits, labels):
    """softmax_cross_entropy."""
    import zero_jax.nn as nn

    return -jnp.sum(labels * nn.log_softmax(logits), axis=-1)
