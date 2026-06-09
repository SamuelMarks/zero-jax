"""Tests for zero_optax.losses."""

import numpy as np
from zero_optax import losses


def test_ctc_losses():
    """Test CTC losses."""
    logits = np.zeros((2, 5, 10))
    paddings = np.zeros((2, 5))
    labels = np.zeros((2, 3))
    label_paddings = np.zeros((2, 3))

    loss1 = losses.ctc_loss(logits, paddings, labels, label_paddings)
    assert loss1.shape == (2,)

    loss2, alpha, loglik = losses.ctc_loss_with_forward_probs(
        logits, paddings, labels, label_paddings
    )
    assert loss2.shape == (2,)
    assert alpha.shape == (2, 5, 2 * 3 + 1)
    assert loglik.shape == loss1.shape


def test_basic_losses():
    """Test basic losses."""
    preds = np.array([0.1, 0.9])
    targets = np.array([0.0, 1.0])

    assert losses.hinge_loss(preds, targets).shape == preds.shape
    assert losses.huber_loss(preds, targets).shape == preds.shape
    assert losses.huber_loss(preds).shape == preds.shape
    assert losses.l2_loss(preds, targets).shape == preds.shape
    assert losses.l2_loss(preds).shape == preds.shape
    assert losses.squared_error(preds, targets).shape == preds.shape
    assert losses.squared_error(preds).shape == preds.shape
    assert losses.perceptron_loss(preds, targets).shape == preds.shape


def test_multiclass_losses():
    """Test multiclass losses."""
    scores = np.array([[0.1, 0.9], [0.8, 0.2]])
    labels = np.array([[0, 1], [1, 0]])

    assert losses.multiclass_hinge_loss(scores, labels).shape == (2,)
    assert losses.multiclass_perceptron_loss(scores, labels).shape == (2,)
    assert losses.multiclass_sparsemax_loss(scores, labels).shape == (2,)
    assert losses.sparsemax_loss(scores, labels).shape == (2, 2)


def test_softmax_cross_entropy():
    """Test softmax cross entropy and its variants."""
    logits = np.array([[0.1, 0.9], [0.8, 0.2]])
    labels = np.array([[0.0, 1.0], [1.0, 0.0]])

    ce = losses.softmax_cross_entropy(logits, labels)
    assert ce.shape == (2,)

    ce_safe = losses.safe_softmax_cross_entropy(logits, labels)
    assert np.allclose(ce, ce_safe)

    int_labels = np.array([1, 0])
    ce_int = losses.softmax_cross_entropy_with_integer_labels(logits, int_labels)
    assert ce_int.shape == (2,)

    # Test with where mask
    ce_mask = losses.softmax_cross_entropy(
        logits, labels, where=np.array([True, False])
    )
    assert ce_mask[1] == 0.0


def test_other_losses():
    """Test other specialized losses."""
    logits = np.array([[0.1, 0.9], [0.8, 0.2]])
    labels = np.array([[0.0, 1.0], [1.0, 0.0]])

    plce = losses.poly_loss_cross_entropy(logits, labels)
    assert plce.shape == (2,)

    rsce = losses.ranking_softmax_loss(logits, labels, weights=np.array([1.0, 0.5]))
    assert rsce.shape == ()  # Reduced by mean

    rsce_none = losses.ranking_softmax_loss(logits, labels, reduce_fn=None)
    assert rsce_none.shape == (2,)


def test_sigmoid_losses():
    """Test sigmoid based losses."""
    logits = np.array([0.1, 0.9])
    labels = np.array([0.0, 1.0])

    bce = losses.sigmoid_binary_cross_entropy(logits, labels)
    assert bce.shape == (2,)

    fl = losses.sigmoid_focal_loss(logits, labels, alpha=0.5)
    assert fl.shape == (2,)


def test_make_fenchel_young_loss():
    """Test make_fenchel_young_loss."""

    def max_fun(x):
        return np.max(x, axis=-1)

    fy_loss_fn = losses.make_fenchel_young_loss(max_fun)
    scores = np.array([[0.1, 0.9], [0.8, 0.2]])
    targets = np.array([[0, 1], [1, 0]])

    loss = fy_loss_fn(scores, targets)
    assert loss.shape == (2,)


def test_losses_softmax():
    """Test softmax with where mask."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = losses.softmax(x, where=np.array([[True, False], [False, True]]))
    assert y[0, 1] == 0.0
    assert y[1, 0] == 0.0


def test_multiclass_sparsemax_loss():
    """Test multiclass sparsemax loss."""
    scores = np.array([[0.1, 0.9], [0.8, 0.2]])
    labels = np.array([[0.0, 1.0], [1.0, 0.0]])
    loss = losses.multiclass_sparsemax_loss(scores, labels)
    assert loss.shape == (2,)


def test_ctc_loss_edge_cases():
    """Test CTC edge cases."""
    # Test blank seq vs valid
    logits = np.zeros((1, 3, 2))
    logits[0, 0, 1] = 10.0  # B, T, V
    paddings = np.array([[0.0, 0.0, 0.0]])
    labels = np.array([[1]])
    label_paddings = np.array([[0.0]])

    loss = losses.ctc_loss(logits, paddings, labels, label_paddings)
    assert loss.shape == (1,)

    # Empty labels
    labels_empty = np.array([[0]])
    label_paddings_empty = np.array([[1.0]])
    loss_empty = losses.ctc_loss(logits, paddings, labels_empty, label_paddings_empty)
    assert loss_empty.shape == (1,)


def test_ctc_loss_more_branches():
    """Test CTC branches."""
    logits = np.zeros((1, 4, 3))
    logits[0, :, :] = 1.0  # B, T, V
    paddings = np.array([[0.0, 0.0, 0.0, 0.0]])
    labels = np.array([[1, 2]])
    label_paddings = np.array([[0.0, 0.0]])

    # Needs a long enough sequence to hit the u-2 logic (character before blank)
    loss, alpha, loglik = losses.ctc_loss_with_forward_probs(
        logits, paddings, labels, label_paddings, log_epsilon=0.0
    )  # force log_epsilon branch
    assert loss.shape == (1,)


def test_ctc_loss_final_loglik():
    """Test CTC log_epsilon branch for final loglik."""
    # A setup where the probability is zero so loglik hits log_epsilon branch
    logits = np.zeros((1, 1, 3))  # B, T, V
    paddings = np.array([[0.0]])
    labels = np.array([[1, 2]])
    label_paddings = np.array([[0.0, 0.0]])

    # Needs to transition through 2 labels in 1 timestep, which is impossible.
    # Therefore, final alpha at those indices remains log_epsilon.
    loss, alpha, loglik = losses.ctc_loss_with_forward_probs(
        logits, paddings, labels, label_paddings, log_epsilon=-1000.0
    )
    assert loglik[0] == -1000.0
