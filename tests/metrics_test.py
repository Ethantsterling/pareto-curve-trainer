"""Tests for metrics.py."""
# pylint: disable=missing-function-docstring
# pylint: disable=wrong-import-position
# For most tests, it's quicker and cleaner to lock JAX to CPU.
import math
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
from pareto_curve_trainer import metrics

###################
# Pointwise Metrics #
###################
def test_mse_fn():
    """Validate our mse_fn."""
    # 0D
    assert metrics.mse_fn(1., 2.) == 1.
    assert metrics.mse_fn(10., 20.) == 100.

    # 1D
    ground_truth_1d = jnp.array([2., 20., 200.])
    pred_1d = jnp.array([1., 10., 100.])
    expected_mse_1d = -(1**2 + 10**2 + 100**2) / 3
    assert jnp.allclose(metrics.mse_fn(ground_truth_1d, pred_1d), expected_mse_1d)
    assert jnp.allclose(metrics.mse_fn(-ground_truth_1d, -pred_1d), expected_mse_1d)

    # 2D
    ground_truth_2d = jnp.array([[2., 20., -200], [5., 3., 2.]])
    pred_2d = jnp.array([[1., 10., -100], [4., 5., 2.]])
    expected_mse_2d = jnp.array([-(1**2 + 10**2 + 100**2) / 3, -(1**2 + 2**2 + 0**2) / 3])
    assert jnp.allclose(metrics.mse_fn(ground_truth_2d, pred_2d), expected_mse_2d)


# For use as a baseline.
def naive_logloss_fn(label: jnp.ndarray, pred: jnp.ndarray):
    """Logloss function that's susceptible to numerical instability."""
    p = jax.nn.sigmoid(pred)  # Convert to probability
    loss = (label * jnp.log(p) + (1 - label) * jnp.log(1 - p))
    if jnp.ndim(pred) == 0:
        return loss
    return jnp.average(loss, axis=-1)


def test_logloss_fn():
    """For reasonably-sized logits, the naive construction is sufficiently accurate."""
    ground_truth = jax.random.uniform(shape=1000, minval=0, maxval=1, key=jax.random.key(5822))
    logits = jax.random.uniform(shape=1000, minval=-10, maxval=10, key=jax.random.key(5823))
    assert jnp.isclose(naive_logloss_fn(ground_truth, logits),
                       metrics.logloss_fn(ground_truth, logits))


def test_logloss_fn_in_two_dimensions():
    """For reasonably-sized logits, the naive construction is sufficiently accurate."""
    ground_truth = jax.random.uniform(shape=(3, 1000), minval=0, maxval=1, key=jax.random.key(5822))
    logits = jax.random.uniform(shape=(3, 1000), minval=-10, maxval=10, key=jax.random.key(5823))

    loss_1, loss_2, loss_3 = metrics.logloss_fn(ground_truth, logits)
    assert jnp.isclose(loss_1, metrics.logloss_fn(ground_truth[0], logits[0]))
    assert jnp.isclose(loss_2, metrics.logloss_fn(ground_truth[1], logits[1]))
    assert jnp.isclose(loss_3, metrics.logloss_fn(ground_truth[2], logits[2]))


def test_logloss_fn_symmetry():
    """Verify that logloss_fn(logit, label) == logloss_fn(-logit, 1 - label)."""
    ground_truth = jax.random.uniform(shape=1000, minval=0, maxval=1, key=jax.random.key(5822))
    logits = jax.random.uniform(shape=1000, minval=-10, maxval=10, key=jax.random.key(5823))
    assert jnp.isclose(metrics.logloss_fn(ground_truth, logits),
                       metrics.logloss_fn(1 - ground_truth, -logits))


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))  # jnp instead of math so that it's float32.


def test_logloss_fn_extremes():
    """Verify that logloss_fn avoids numerical instability at the extremes.
    
    Logloss relies on log and exp in ways that are very sensitive to float imprecision.
    We wrote logloss_fn carefully to minimize that sensitivity."""
    # label = 0 --> -jnp.log(1 - sigmoid(logit))
    # label = 1 --> -jnp.log(sigmoid(logit))

    # Confidently-wrong predictions should approach -logit.
    assert jnp.isclose(metrics.logloss_fn(0, 100), -100)
    assert jnp.isclose(metrics.logloss_fn(1, -100), -100)
    assert jnp.isclose(metrics.logloss_fn(0, 40), -40)
    assert jnp.isclose(metrics.logloss_fn(1, -40), -40)

    # Confidently-right predictions should approach -sigmoid(-abs(logit)).
    assert jnp.isclose(metrics.logloss_fn(1, 100), -sigmoid(-100))
    assert jnp.isclose(metrics.logloss_fn(0, -100), -sigmoid(-100))
    assert jnp.isclose(metrics.logloss_fn(1, 40), -sigmoid(-40))
    assert jnp.isclose(metrics.logloss_fn(0, -40), -sigmoid(-40))

###################
# Ranking Metrics #
###################


def dcg_group_scores(ordered_labels, ordered_groups, depth=None):
    """Calculate the per-group scores for distributed cumulative gain."""
    group_gain = []
    prev_group = None
    in_group_rank = 1  # nDCG considers the first rank in a group to be 1.
    for group, label in zip(ordered_groups.tolist(), ordered_labels.tolist()):
        if prev_group != group:
            in_group_rank = 1
            group_gain.append(0)
        else:
            in_group_rank += 1
        prev_group = group
        if depth is not None and in_group_rank > depth:
            continue
        gain_discount = 1 / math.log2(in_group_rank + 1)
        group_gain[-1] += label * gain_discount
    return group_gain


def dcg_metric(score, labels, groups, depth=None) -> float:
    """Distributed Cumulative Gain metric, sorting by `score`.
    
    This function is a baseline for testing. It's written in simple, inefficient code to ensure
    it's obviously correct."""
    order = jnp.lexsort([-score, groups])  # Higher score --> earlier.
    ordered_groups = groups[order]
    ordered_labels = labels[order]
    group_scores = dcg_group_scores(ordered_labels, ordered_groups, depth)
    return sum(group_scores) / len(group_scores)


def ndcg_metric(score, labels, groups, depth=None):
    """Normalized Distributed Cumulative Gain metric, sorting by `score`.
    
    This function is a baseline for testing. It's written in simple, inefficient code to ensure
    it's obviously correct."""
    order = jnp.lexsort([-score, groups])  # Higher score --> earlier.
    ideal_order = jnp.lexsort([-labels, groups])  # Higher score --> earlier.
    group_scores = jnp.array(dcg_group_scores(labels[order], groups[order], depth))
    ideal_group_scores = jnp.array(
        dcg_group_scores(labels[ideal_order], groups[ideal_order], depth))
    ratios = group_scores / ideal_group_scores
    return jnp.average(ratios)


def test_dcg_test_functions():
    # https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Example
    labels = jnp.array([3, 2, 3, 0, 1, 2.])
    groups = jnp.array([0, 0, 0, 0, 0, 0])
    group_scores = dcg_group_scores(labels, groups)
    assert len(group_scores) == 1  # Since there's only one group.
    assert jnp.isclose(group_scores[0], 6.861, atol=1e-3)

    score = -jnp.arange(len(labels))  # To keep the sorting in the current order.
    dcg_score = dcg_metric(score, labels, groups)
    assert jnp.isclose(dcg_score, 6.861, atol=1e-3)

    ideal_score = labels
    ideal_dcg_score = dcg_metric(ideal_score, labels, groups)
    # assert jnp.isclose(ideal_dcg_score, 8.740, atol=1e-3)

    ndcg_score = ndcg_metric(score, labels, groups)
    assert jnp.isclose(ndcg_score, dcg_score / ideal_dcg_score, atol=1e-3)


def test_dcg_test_functions_with_depth():
    # https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Example
    labels = jnp.array([3, 2, 3, 0, 1, 2, 3, 2.])
    groups = jnp.array([0, 0, 0, 0, 0, 0, 0, 0])
    group_scores = dcg_group_scores(labels, groups, depth=6)
    assert len(group_scores) == 1  # Since there's only one group.
    assert jnp.isclose(group_scores[0], 6.861, atol=1e-3)

    score = -jnp.arange(len(labels))  # To keep the sorting in the current order.
    dcg_score = dcg_metric(score, labels, groups, depth=6)
    assert jnp.isclose(dcg_score, 6.861, atol=1e-3)

    ideal_score = labels
    ideal_dcg_score = dcg_metric(ideal_score, labels, groups, depth=6)
    assert jnp.isclose(ideal_dcg_score, 8.740, atol=1e-3)

    ndcg_score = ndcg_metric(score, labels, groups, depth=6)
    assert jnp.isclose(ndcg_score, dcg_score / ideal_dcg_score, atol=1e-3)


def test_dcg_test_functions_with_groups():
    # https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Example
    labels = jnp.array([3, 2, 3, 0, 1, 2, 3, 2., 3, 2, 3, 0, 1, 2,])
    groups = jnp.array([0, 0, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1])
    group_scores = dcg_group_scores(labels, groups, depth=6)
    assert len(group_scores) == 2
    assert jnp.allclose(jnp.array(group_scores), jnp.array([6.861, 6.861]), atol=1e-3)

    score = -jnp.arange(len(labels))  # To keep the sorting in the current order.
    dcg_score = dcg_metric(score, labels, groups, depth=6)
    assert jnp.isclose(dcg_score, 6.861, atol=1e-3)

    ideal_order = jnp.lexsort([-labels, groups])
    ideal_group_scores = dcg_group_scores(labels[ideal_order], groups[ideal_order], depth=6)

    expected_ndcg_score = (group_scores[0] / ideal_group_scores[0] +
                           group_scores[1] / ideal_group_scores[1]) / 2
    ndcg_score = ndcg_metric(score, labels, groups, depth=6)
    assert jnp.isclose(ndcg_score, expected_ndcg_score, atol=1e-3)


def test_anchored_ndcg_metric():
    labels = jnp.array([3, 2, 3, 0, 1, 2, 3, 2.])
    groups = jnp.array([0, 0, 0, 0, 0, 0, 0, 0])

    scores = -jnp.arange(len(labels))  # To keep the sorting in the current order.
    ndcg_score = ndcg_metric(scores, labels, groups)

    order = jnp.lexsort([-scores, groups])
    anchored_ndcg = metrics.anchored_ndcg_metric(labels, groups)
    anchored_ndcg_score = anchored_ndcg.eval(labels[order])
    assert jnp.isclose(ndcg_score, anchored_ndcg_score)

    # With depth=5.
    ndcg_score = ndcg_metric(scores, labels, groups, depth=5)
    anchored_ndcg = metrics.anchored_ndcg_metric(labels, groups, depth=5)
    anchored_ndcg_score = anchored_ndcg.eval(labels[order])
    assert jnp.isclose(ndcg_score, anchored_ndcg_score)


def test_anchored_ndcg_metric_on_groups():
    labels = jax.random.uniform(shape=(100,), key=jax.random.key(1))
    groups = jax.random.randint(shape=(100,), minval=1, maxval=4, key=jax.random.key(2))
    scores = jax.random.uniform(shape=(100,), key=jax.random.key(3))
    # Shift the scores to correlate somewhat with labels so they're not just random.
    scores = scores + .2 * labels

    # depth=None
    ndcg_score = ndcg_metric(scores, labels, groups)
    anchored_ndcg = metrics.anchored_ndcg_metric(labels, groups)
    order = jnp.lexsort([-scores, groups])
    anchored_ndcg_score = anchored_ndcg.eval(labels[order])
    assert jnp.isclose(ndcg_score, anchored_ndcg_score, atol=1e-5)

    perfect_order = jnp.lexsort([-labels, groups])
    anchored_ndcg_score = anchored_ndcg.eval(labels[perfect_order])
    assert jnp.isclose(anchored_ndcg_score, 1, atol=1e-5)

    # Reverse score order to test a bad order.
    ndcg_score = ndcg_metric(-scores, labels, groups)
    anchored_ndcg = metrics.anchored_ndcg_metric(labels, groups)
    order = jnp.lexsort([scores, groups])
    anchored_ndcg_score = anchored_ndcg.eval(labels[order])
    assert jnp.isclose(ndcg_score, anchored_ndcg_score, atol=1e-5)

    # depth=5
    ndcg_score = ndcg_metric(scores, labels, groups, depth=5)
    anchored_ndcg = metrics.anchored_ndcg_metric(labels, groups, depth=5)
    order = jnp.lexsort([-scores, groups])
    anchored_ndcg_score = anchored_ndcg.eval(labels[order])
    assert jnp.isclose(ndcg_score, anchored_ndcg_score, atol=1e-5)

    anchored_ndcg = metrics.anchored_ndcg_metric(labels, groups, depth=5)
    perfect_order = jnp.lexsort([-labels, groups])
    anchored_ndcg_score = anchored_ndcg.eval(labels[perfect_order])
    assert jnp.isclose(anchored_ndcg_score, 1, atol=1e-5)

    # Reverse score order to test a bad order.
    ndcg_score = ndcg_metric(-scores, labels, groups, depth=5)
    anchored_ndcg = metrics.anchored_ndcg_metric(labels, groups, depth=5)
    order = jnp.lexsort([scores, groups])
    anchored_ndcg_score = anchored_ndcg.eval(labels[order])
    assert jnp.isclose(ndcg_score, anchored_ndcg_score, atol=1e-5)


def test_anchored_linear_metric():
    labels = jax.random.uniform(shape=(100,), key=jax.random.key(1))
    groups = jax.random.randint(shape=(100,), minval=1, maxval=4, key=jax.random.key(2))
    scores = jax.random.uniform(shape=(100,), key=jax.random.key(3)) + .2 * labels

    # For now just test that it runs.
    anchored_linear = metrics.anchored_linear_metric(labels, groups)
    anchored_linear.eval(scores)
    anchored_linear = metrics.anchored_linear_metric(labels)
    anchored_linear.eval(scores)
    anchored_linear = metrics.anchored_linear_metric(labels, groups, depth=5)
    anchored_linear.eval(scores)


def test_group_rank():
    labels = jax.random.normal(shape=1000, key=jax.random.key(50))
    groups = jax.random.randint(shape=1000, minval=0, maxval=4, key=jax.random.key(51))
    assert jnp.all(metrics.group_rank(labels, groups) == jnp.lexsort([-labels, groups]))

    # We'd prefer that the tiebreaking behavior also matches.
    labels_with_ties = jax.random.randint(shape=1000, minval=0, maxval=5, key=jax.random.key(50))
    assert jnp.all(metrics.group_rank(labels_with_ties, groups) ==
                   jnp.lexsort([-labels_with_ties, groups]))
