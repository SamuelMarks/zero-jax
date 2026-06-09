"""Optax losses.

This module implements various loss functions.
"""

from typing import (
    Any,
    Callable,
    Union,
    Optional,
    Tuple,
    TypeVar,
)
import numpy as np

Array = Any
Numeric = Any


class AxisName:
    """Docstring."""

    pass


class PrecisionLike:
    """Docstring."""

    pass


class PaddingLike:
    """Docstring."""

    pass


class Dtype:
    """Docstring."""

    pass


class Shape:
    """Docstring."""

    pass


class Axes:
    """Docstring."""

    pass


class Size:
    """Docstring."""

    pass


class Axis:
    """Docstring."""

    pass


class DotGeneralT:
    """Docstring."""

    pass


class MaxFun:
    """Docstring."""

    pass


class filterlib:
    """Docstring."""

    class Filter:
        """Docstring."""

        pass


class rnglib:
    """Docstring."""

    class Rngs:
        """Docstring."""

        pass


class variables:
    """Docstring."""

    class Variable:
        """Docstring."""

        pass


class chex:
    """Docstring."""

    class Array:
        """Docstring."""

        pass

    class Numeric:
        """Docstring."""

        pass

    class Scalar:
        """Docstring."""

        pass


class core:
    """Docstring."""

    class Shape:
        """Docstring."""

        pass


class optax:
    """Docstring."""

    class _src:
        """Docstring."""

        class base:
            """Docstring."""

            class GradientTransformationExtraArgs:
                """Docstring."""

                pass


class base:
    """Docstring."""

    class GradientTransformation:
        """Docstring."""

        pass

    class Schedule:
        """Docstring."""

        pass


class jax:
    """Docstring."""

    class Array:
        """Docstring."""

        pass

    class Device:
        """Docstring."""

        pass

    class _src:
        """Docstring."""

        class typing:
            """Docstring."""

            class SupportsDType:
                """Docstring."""

                pass


M = TypeVar("M") if "TypeVar" in globals() else Any
A = TypeVar("A") if "TypeVar" in globals() else Any
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


def ctc_loss(
    logits: Array,
    logit_paddings: Array,
    labels: Array,
    label_paddings: Array,
    blank_id: int = 0,
    log_epsilon: float = -100000.0,
) -> Array:
    """Computes CTC loss.

    Args:
        logits: Logits.
        logit_paddings: Paddings for logits.
        labels: Labels.
        label_paddings: Paddings for labels.
        blank_id: ID of the blank token.
        log_epsilon: Epsilon for log calculations.

    Returns:
        CTC loss.
    """
    loss, _, _ = ctc_loss_with_forward_probs(
        logits, logit_paddings, labels, label_paddings, blank_id, log_epsilon
    )
    return loss


def ctc_loss_with_forward_probs(
    logits: Array,
    logit_paddings: Array,
    labels: Array,
    label_paddings: Array,
    blank_id: int = 0,
    log_epsilon: float = -100000.0,
) -> Tuple[Array, Array, Array]:
    """Computes CTC loss and CTC forward-probabilities.

    Args:
        logits: Logits.
        logit_paddings: Paddings for logits.
        labels: Labels.
        label_paddings: Paddings for labels.
        blank_id: ID of the blank token.
        log_epsilon: Epsilon for log calculations.

    Returns:
        A tuple of (CTC loss, alpha, log_likelihood).
    """
    B, T, V = logits.shape
    logits = np.where(np.expand_dims(logit_paddings, axis=-1) == 1, log_epsilon, logits)

    # Compute log probabilities over vocabulary
    log_probs = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))

    L = labels.shape[1]
    U = 2 * L + 1

    # Prepare labels with blanks
    # blank sequence: b l1 b l2 b ... b
    # e.g. [0, label[0], 0, label[1], 0]
    padded_labels = np.full((B, U), blank_id, dtype=np.int32)
    padded_labels[:, 1::2] = labels

    alpha = np.full((B, T, U), log_epsilon, dtype=np.float32)

    for b in range(B):
        # Initialize alpha
        alpha[b, 0, 0] = log_probs[b, 0, blank_id]
        if L > 0:
            l0 = int(labels[b, 0])
            alpha[b, 0, 1] = log_probs[b, 0, l0]

        seq_len = int(T - np.sum(logit_paddings[b]))
        lab_len = int(L - np.sum(label_paddings[b]))
        max_u = 2 * lab_len + 1

        for t in range(1, seq_len):
            for u in range(max_u):
                curr_char = int(padded_labels[b, u])

                # Transition from same character
                term1 = alpha[b, t - 1, u]

                # Transition from previous character
                term2 = alpha[b, t - 1, u - 1] if u > 0 else log_epsilon

                # Transition from character before blank (if not blank and not same as previous)
                term3 = log_epsilon
                if (
                    u > 1
                    and curr_char != blank_id
                    and curr_char != padded_labels[b, u - 2]
                ):
                    term3 = alpha[b, t - 1, u - 2]

                max_prev = np.maximum(np.maximum(term1, term2), term3)
                if max_prev == log_epsilon:
                    sum_prev = log_epsilon
                else:
                    sum_prev = max_prev + np.log(
                        np.exp(term1 - max_prev)
                        + np.exp(term2 - max_prev)
                        + np.exp(term3 - max_prev)
                    )

                alpha[b, t, u] = sum_prev + log_probs[b, t, curr_char]

    loss = np.zeros(B, dtype=logits.dtype)
    log_likelihood = np.zeros(B, dtype=logits.dtype)

    for b in range(B):
        seq_len = int(T - np.sum(logit_paddings[b]))
        lab_len = int(L - np.sum(label_paddings[b]))

        if lab_len == 0:
            loglik = alpha[b, seq_len - 1, 0]
        else:
            idx1 = 2 * lab_len
            idx2 = 2 * lab_len - 1
            val1 = alpha[b, seq_len - 1, idx1]
            val2 = alpha[b, seq_len - 1, idx2]

            max_val = max(val1, val2)
            if max_val == log_epsilon:
                loglik = log_epsilon
            else:
                loglik = max_val + np.log(
                    np.exp(val1 - max_val) + np.exp(val2 - max_val)
                )

        log_likelihood[b] = loglik
        loss[b] = -loglik

    return loss, alpha, log_likelihood


def hinge_loss(predictor_outputs: Array, targets: Array) -> Array:
    """Computes the hinge loss for binary classification.

    Args:
        predictor_outputs: Predictions.
        targets: Target labels.

    Returns:
        Hinge loss.
    """
    return np.maximum(0.0, 1.0 - predictor_outputs * targets)


def huber_loss(
    predictions: Array, targets: Optional[Array] = None, delta: float = 1.0
) -> Array:
    """Huber loss, similar to L2 loss close to zero, L1 loss away from zero.

    Args:
        predictions: Predictions.
        targets: Targets.
        delta: Delta parameter.

    Returns:
        Huber loss.
    """
    if targets is None:
        targets = np.zeros_like(predictions)
    abs_err = np.abs(predictions - targets)
    return np.where(abs_err < delta, 0.5 * abs_err**2, delta * (abs_err - 0.5 * delta))


def l2_loss(predictions: Array, targets: Optional[Array] = None) -> Array:
    """Calculates the L2 loss for a set of predictions.

    Args:
        predictions: Predictions.
        targets: Targets.

    Returns:
        L2 loss.
    """
    if targets is None:
        targets = np.zeros_like(predictions)
    return 0.5 * (predictions - targets) ** 2


def make_fenchel_young_loss(max_fun: Callable[..., Any]) -> Callable[..., Any]:
    """Creates a Fenchel-Young loss from a max function.

    Args:
        max_fun: A max function.

    Returns:
        A Fenchel-Young loss function.
    """

    def fy_loss(scores: Array, targets: Array) -> Array:
        """Docstring."""
        return np.maximum(0.0, max_fun(scores) - np.sum(scores * targets, axis=-1))

    return fy_loss


def multiclass_hinge_loss(scores: Array, labels: Array) -> Array:
    """Multiclass hinge loss.

    Args:
        scores: Scores.
        labels: Labels.

    Returns:
        Multiclass hinge loss.
    """
    # Assuming labels are one-hot
    true_scores = np.sum(scores * labels, axis=-1, keepdims=True)
    margins = np.maximum(0.0, 1.0 - true_scores + scores)
    margins = margins * (1.0 - labels)
    return np.max(margins, axis=-1)


def multiclass_perceptron_loss(scores: Array, labels: Array) -> Array:
    """Multiclass perceptron loss.

    Args:
        scores: Scores.
        labels: Labels.

    Returns:
        Multiclass perceptron loss.
    """
    true_scores = np.sum(scores * labels, axis=-1, keepdims=True)
    return np.maximum(
        0.0, np.max(scores, axis=-1, keepdims=True) - true_scores
    ).squeeze(-1)


def _sparsemax_projection(z: Array) -> Array:
    """Computes the sparsemax projection."""
    original_shape = z.shape
    z_flat = z.reshape(-1, z.shape[-1])
    sorted_z = np.sort(z_flat, axis=-1)[:, ::-1]

    cumsum_z = np.cumsum(sorted_z, axis=-1)
    k = np.arange(1, z.shape[-1] + 1)
    is_gt = sorted_z * k > (cumsum_z - 1)
    k_max = np.sum(is_gt, axis=-1, keepdims=True)

    # Compute threshold tau
    tau = (np.take_along_axis(cumsum_z, k_max - 1, axis=-1) - 1) / k_max

    p = np.maximum(0.0, z_flat - tau)
    return p.reshape(original_shape)


def multiclass_sparsemax_loss(scores: Array, labels: Array) -> Array:
    """Multiclass sparsemax loss.

    Args:
        scores: Scores.
        labels: Labels.

    Returns:
        Multiclass sparsemax loss.
    """
    p = _sparsemax_projection(scores)

    # 0.5 * ||p - labels||^2 - 0.5 * ||p||^2 + p \dot scores - labels \dot scores
    # = 0.5 * ||labels||^2 - labels \dot scores + p \dot scores - 0.5 * ||p||^2
    # Assuming labels are one-hot, ||labels||^2 = 1

    loss = (
        0.5 * np.sum(labels**2, axis=-1)
        - np.sum(labels * scores, axis=-1)
        + np.sum(p * scores, axis=-1)
        - 0.5 * np.sum(p**2, axis=-1)
    )
    return loss


def perceptron_loss(predictor_outputs: Numeric, targets: Numeric) -> Numeric:
    """Binary perceptron loss.

    Args:
        predictor_outputs: Predictions.
        targets: Targets.

    Returns:
        Perceptron loss.
    """
    return np.maximum(0.0, -predictor_outputs * targets)


def poly_loss_cross_entropy(
    logits: Array,
    labels: Array,
    epsilon: float = 2.0,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
    where: Optional[Array] = None,
) -> Array:
    """Computes PolyLoss between logits and labels.

    Args:
        logits: Logits.
        labels: Labels.
        epsilon: Epsilon parameter.
        axis: Axis for cross entropy.
        where: Elements to include.

    Returns:
        PolyLoss.
    """
    ce = softmax_cross_entropy(logits, labels, axis=axis, where=where)
    pt = np.sum(labels * softmax(logits, axis=axis), axis=axis)
    return ce + epsilon * (1.0 - pt)


def ranking_softmax_loss(
    logits: Array,
    labels: Array,
    where: Optional[Array] = None,
    weights: Optional[Array] = None,
    reduce_fn: Optional[Callable[..., Array]] = np.mean,
) -> Array:
    """Ranking softmax loss.

    Args:
        logits: Logits.
        labels: Labels.
        where: Elements to include.
        weights: Weights.
        reduce_fn: Reduction function.

    Returns:
        Ranking softmax loss.
    """
    loss = softmax_cross_entropy(logits, labels, where=where)
    if weights is not None:
        loss = loss * weights
    if reduce_fn is not None:
        return reduce_fn(loss)
    return loss


def safe_softmax_cross_entropy(logits: Array, labels: Array) -> Array:
    """Computes the softmax cross entropy between sets of logits and labels safely.

    Args:
        logits: Logits.
        labels: Labels.

    Returns:
        Softmax cross entropy.
    """
    return softmax_cross_entropy(logits, labels)


def sigmoid_binary_cross_entropy(logits: Array, labels: Array) -> Array:
    """Computes element-wise sigmoid cross entropy given logits and labels.

    Args:
        logits: Logits.
        labels: Labels.

    Returns:
        Sigmoid binary cross entropy.
    """
    log_p = -np.logaddexp(0.0, -logits)
    log_not_p = -np.logaddexp(0.0, logits)
    return -(labels * log_p + (1.0 - labels) * log_not_p)


def sigmoid_focal_loss(
    logits: Array, labels: Array, alpha: Optional[float] = None, gamma: float = 2.0
) -> Array:
    """Sigmoid focal loss.

    Args:
        logits: Logits.
        labels: Labels.
        alpha: Alpha parameter.
        gamma: Gamma parameter.

    Returns:
        Sigmoid focal loss.
    """
    p = 1.0 / (1.0 + np.exp(-logits))
    ce = sigmoid_binary_cross_entropy(logits, labels)
    p_t = p * labels + (1.0 - p) * (1.0 - labels)
    loss = ce * ((1.0 - p_t) ** gamma)
    if alpha is not None:
        alpha_t = alpha * labels + (1.0 - alpha) * (1.0 - labels)
        loss = alpha_t * loss
    return loss


def softmax(
    x: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
    where: Optional[Array] = None,
) -> Array:
    """Softmax function.

    Args:
        x: Input array.
        axis: Axis.
        where: Mask.

    Returns:
        Softmax activation.
    """
    x = np.asarray(x)
    if where is not None:
        x_max = np.max(np.where(where, x, -np.inf), axis=axis, keepdims=True)
    else:
        x_max = np.max(x, axis=axis, keepdims=True)
    unnorm = np.exp(x - x_max)
    if where is not None:
        unnorm = np.where(where, unnorm, 0.0)
    return unnorm / np.sum(unnorm, axis=axis, keepdims=True)


def softmax_cross_entropy(
    logits: Array,
    labels: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
    where: Optional[Array] = None,
) -> Array:
    """Computes the softmax cross entropy between sets of logits and labels.

    Args:
        logits: Logits.
        labels: Labels.
        axis: Axis.
        where: Mask.

    Returns:
        Softmax cross entropy.
    """
    x = np.asarray(logits)
    x_max = np.max(x, axis=axis, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    log_p = x - log_sum_exp
    loss = -np.sum(labels * log_p, axis=axis)
    if where is not None:
        loss = np.where(where, loss, 0.0)
    return loss


def softmax_cross_entropy_with_integer_labels(
    logits: Array,
    labels: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
    where: Optional[Array] = None,
) -> Array:
    """Computes softmax cross entropy between the logits and integer labels.

    Args:
        logits: Logits.
        labels: Labels.
        axis: Axis.
        where: Mask.

    Returns:
        Softmax cross entropy.
    """
    # Convert integer labels to one-hot manually
    num_classes = logits.shape[axis if axis is not None else -1]
    one_hot_labels = np.eye(num_classes)[labels]
    return softmax_cross_entropy(logits, one_hot_labels, axis=axis, where=where)


def sparsemax_loss(logits: Array, labels: Array) -> Array:
    """Binary sparsemax loss.

    Args:
        logits: Logits.
        labels: Labels.

    Returns:
        Sparsemax loss.
    """
    p = np.clip(0.5 * logits + 0.5, 0.0, 1.0)
    loss = -labels * p + 0.5 * (p**2 + labels**2)
    return loss


def squared_error(predictions: Array, targets: Optional[Array] = None) -> Array:
    """Calculates the squared error for a set of predictions.

    Args:
        predictions: Predictions.
        targets: Targets.

    Returns:
        Squared error.
    """
    if targets is None:
        targets = np.zeros_like(predictions)
    return (predictions - targets) ** 2
