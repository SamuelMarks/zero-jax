from zero_jax import numpy as jnp


def l2_loss(predictions, targets):
    return jnp.multiply(0.5, jnp.power(jnp.subtract(predictions, targets), 2))


def softmax_cross_entropy(logits, labels):
    import zero_jax.nn as nn

    return jnp.negative(jnp.sum(jnp.multiply(labels, nn.log_softmax(logits)), axis=-1))
