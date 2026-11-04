"""Tests for metrics_util.py."""
# pylint: disable=missing-function-docstring
# pylint: disable=wrong-import-position
# For most tests, it's quicker and cleaner to lock JAX to CPU.
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import pareto_curve_trainer.metrics_util as mu


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))  # jnp instead of math so that it's float32.


def test_log1p_negative_sigmoid():
    normal_xs = jnp.linspace(start=5, stop=-5)
    assert jnp.allclose(
        mu.log1p_negative_sigmoid(normal_xs),
        jnp.log(1 - sigmoid(normal_xs)))

    # For extreme xs, verify that we don't overflow.
    extreme_xs = jnp.linspace(start=40, stop=100)
    assert jnp.allclose(mu.log1p_negative_sigmoid(extreme_xs), -extreme_xs)
    assert jnp.allclose(mu.log1p_negative_sigmoid(-extreme_xs), -sigmoid(-extreme_xs))


def test_streak_lengths():
    a = jnp.array([2., 2., 3., 5., 5., 5., 6., 6., 7, 8.])
    assert jnp.all(mu.streak_lengths(a) == jnp.array([2, 1, 3, 2, 1, 1]))


def test_first_index_matching():
    arr = jnp.array([2., 2., 3., 5., 5., 5., 1., 1., 7, 8.])
    fim  = jnp.array([0,  0,  2,  3,  3,  3,  6,  6,  8, 9])
    assert jnp.all(mu.first_index_matching(arr) == fim)


def eval_order_metric(weights, ordered_labels) -> float:
    """Evaluate an order metric characterized by weights, with labels in the given order."""
    return jnp.average(ordered_labels * weights)


def test_normalize_weights_to_max_one():
    weights = jnp.array([1, .8, .6, .4, .2, 0])
    labels = jnp.array([5., 2., 4., 1., 1., 3.])
    one_weights = mu.normalize_weights_to_max_one(weights, labels)
    ratio = one_weights[0] / weights[0]
    assert jnp.allclose(weights * ratio, one_weights)

    min_score = eval_order_metric(one_weights, labels.sort())
    max_score = eval_order_metric(one_weights, labels.sort(descending=True))

    assert jnp.isclose(max_score, 1.0)
    assert max_score > min_score


def test_normalize_weights_to_max_one_min_zero():
    weights = jnp.array([1, .8, .6, .5, .2, 0])
    labels = jnp.array([5., 2., 4., 1., 1., 3.])
    unit_weights = mu.normalize_weights_to_max_one_min_zero(weights, labels)
    min_score = eval_order_metric(unit_weights, labels.sort())
    max_score = eval_order_metric(unit_weights, labels.sort(descending=True))

    assert jnp.isclose(max_score, 1.0)
    assert jnp.isclose(min_score, 0.0, atol=1e-5)


def test_normalize_weights_to_max_one_grouped():
    weights = jnp.array([1, .8, .6, .4, .2, 0])
    labels = jnp.array([5., 2., 4., 1., 1., 3.])
    groups = jnp.array([3., 2., 2., 1., 1., 4.])
    one_weights = mu.normalize_weights_to_max_one(weights, labels, groups=groups)

    # Normalization should be a scalar multiplication of all the weights.
    ratio = one_weights[0] / weights[0]
    assert jnp.allclose(weights * ratio, one_weights)

    min_score = eval_order_metric(one_weights, labels[jnp.lexsort([labels, groups])])
    max_score = eval_order_metric(one_weights, labels[jnp.lexsort([-labels, groups])])

    assert jnp.isclose(max_score, 1.0)
    assert max_score > min_score


def test_normalize_weights_to_max_one_min_zero_grouped():
    weights = jnp.array([1, .8, .6, .4, .2, 0])
    labels =  jnp.array([5., 2,  4,  1,  1, 3])
    groups =  jnp.array([3,  2,  2,  1,  1, 4])
    one_weights = mu.normalize_weights_to_max_one_min_zero(weights, labels, groups=groups)
    min_score = eval_order_metric(one_weights, labels[jnp.lexsort([labels, groups])])
    max_score = eval_order_metric(one_weights, labels[jnp.lexsort([-labels, groups])])

    assert jnp.isclose(max_score, 1.0)
    assert jnp.isclose(min_score, 0.0, atol=1e-5)


def test_normalize_group_weights_to_max_one():
    weights = jnp.array([1, .8, .6, .4, .2, 0])
    labels =  jnp.array([5., 2,  4,  1,  1, 3])
    groups =  jnp.array([3,  2,  2,  1,  1, 3])
    # Lexsort --> groups [1,1,2,2,3,3]
    #             labels [1,1,4,2,5,3] (for the ideal label order).
    one_weights = mu.normalize_group_weights_to_max_one(weights, labels, groups=groups)

    # Divide each group by its current max score to normalize it to 1.
    expected_normalized_weights = jnp.array([1/.9, .8/.9, .6/1.6, .4/1.6, .2/.5, 0/.5])
    assert jnp.allclose(one_weights, expected_normalized_weights)

    min_score = eval_order_metric(one_weights, labels[jnp.lexsort([labels, groups])])
    max_score = eval_order_metric(one_weights, labels[jnp.lexsort([-labels, groups])])

    assert jnp.isclose(max_score, 1.0)
    assert max_score > min_score


def test_normalize_group_weights_to_max_one_min_zero():
    weights = jnp.array([1, .8, .6, .4, .2, 0])
    labels =  jnp.array([5., 2,  4,  0,  1, 3])
    groups =  jnp.array([3,  2,  2,  1,  1, 3])
    # Lexsort --> groups [1,1,2,2,3,3]
    #             labels [1,0,4,2,5,3] (for the ideal label order).
    one_weights = mu.normalize_group_weights_to_max_one_min_zero(weights, labels, groups=groups)
    # Per-group max scores are currently .5, 1.6, and 0.5.
    # Per-group min scores are currently .4, 1.4, and 0.3.
    # We need to reduce the group weights by .4/.5, 1.4/3, and .3/4.
    # After that shift, the min scores will be 0 and the max scores will be .1, .2, .2.
    # We'll divide through by those max scores.
    min_shift = jnp.array([.4/.5, .4/.5, 1.4/3, 1.4/3, .3/4, .3/4])
    max_divisor = jnp.array([.1, .1, .2, .2, .2, .2])
    expected_normalized_weights = (weights - min_shift) / max_divisor
    assert jnp.allclose(one_weights, expected_normalized_weights)

    min_score = eval_order_metric(one_weights, labels[jnp.lexsort([labels, groups])])
    max_score = eval_order_metric(one_weights, labels[jnp.lexsort([-labels, groups])])
    assert jnp.isclose(max_score, 1.0)
    assert jnp.isclose(min_score, 0.0, atol=1e-5)


def test_within_group_rank():
    groups = jnp.array([3, 2, 2, 1, 1, 1, 3])
    assert jnp.all(mu.within_group_rank(groups) == jnp.array([0, 1, 2, 0, 1, 0, 1]))


def test_order_metric_weights_without_groups():
    weights_arr = jnp.array([1, 1/2, 1/4, 1/8, 1/16])
    def rank_to_weight(rank):
        return (2.0) ** -rank

    labels =  jnp.array([5., 2, 0,  3, 3])
    best_order_labels = labels.sort(descending=True)
    worst_order_labels = labels.sort()
    full_expected_weights = weights_arr

    non_normalized_weights_from_fn = mu.order_metric_weights(
        rank_to_weight, labels, normalize=mu.Normalization.NONE)
    non_normalized_weights_from_arr = mu.order_metric_weights(
        weights_arr, labels, normalize=mu.Normalization.NONE)
    assert jnp.allclose(non_normalized_weights_from_fn, non_normalized_weights_from_arr)
    # Without normalization, the weights are unchanged.
    assert jnp.allclose(non_normalized_weights_from_fn, full_expected_weights)

    unit_normalized_weights_from_fn = mu.order_metric_weights(
        rank_to_weight, labels, normalize=mu.Normalization.UNIT_WEIGHTS)
    unit_normalized_weights_from_arr = mu.order_metric_weights(
        weights_arr, labels, normalize=mu.Normalization.UNIT_WEIGHTS)
    average_weight = jnp.average(full_expected_weights)
    assert jnp.allclose(unit_normalized_weights_from_fn, unit_normalized_weights_from_arr)
    assert jnp.allclose(unit_normalized_weights_from_fn, full_expected_weights / average_weight)

    weights_from_fn = mu.order_metric_weights(
        rank_to_weight, labels, normalize=mu.Normalization.MAX_ONE)
    weights_from_arr = mu.order_metric_weights(
        weights_arr, labels, normalize=mu.Normalization.MAX_ONE)
    assert jnp.allclose(weights_from_fn, weights_from_arr)
    assert jnp.isclose(jnp.average(weights_from_fn * best_order_labels), 1)

    weights_from_fn = mu.order_metric_weights(
        rank_to_weight, labels, normalize=mu.Normalization.MAX_ONE_MIN_ZERO)
    weights_from_arr = mu.order_metric_weights(
        weights_arr, labels, normalize=mu.Normalization.MAX_ONE_MIN_ZERO)
    assert jnp.allclose(weights_from_fn, weights_from_arr)
    assert jnp.isclose(jnp.average(weights_from_fn * best_order_labels), 1)
    assert jnp.isclose(jnp.average(weights_from_fn * worst_order_labels), 0, atol=1e-5)

    # Group normalization without groups is equivalent to non-group normalization.
    assert jnp.allclose(mu.order_metric_weights(
        rank_to_weight, labels, normalize=mu.Normalization.UNIT_WEIGHTS),
                         mu.order_metric_weights(
        rank_to_weight, labels, normalize=mu.Normalization.GROUP_UNIT_WEIGHTS))
    assert jnp.allclose(mu.order_metric_weights(
        rank_to_weight, labels, normalize=mu.Normalization.MAX_ONE),
                         mu.order_metric_weights(
        rank_to_weight, labels, normalize=mu.Normalization.GROUP_MAX_ONE))
    assert jnp.allclose(mu.order_metric_weights(
        rank_to_weight, labels, normalize=mu.Normalization.MAX_ONE_MIN_ZERO),
                         mu.order_metric_weights(
        rank_to_weight, labels, normalize=mu.Normalization.GROUP_MAX_ONE_MIN_ZERO))


def test_order_metric_weights_with_groups():
    weights_arr = jnp.array([1, .5, .25])
    def rank_to_weight(rank):
        return (2.0) ** -rank

    labels =  jnp.array([5., 2,  4,  0,  1, 3, 3])
    groups =  jnp.array([3,  2,  2,  1,  1, 3, 3])
    # [1 0 4 2 5 3 3]
    best_order_labels = labels[jnp.lexsort([-labels, groups])]
    # [0 1 2 4 3 3 5]
    worst_order_labels = labels[jnp.lexsort([labels, groups])]
    full_expected_weights = jnp.array([1, .5, 1, .5, 1, .5, .25])

    non_normalized_weights_from_fn = mu.order_metric_weights(
        rank_to_weight, labels, groups, normalize=mu.Normalization.NONE)
    non_normalized_weights_from_arr = mu.order_metric_weights(
        weights_arr, labels, groups, normalize=mu.Normalization.NONE)
    assert jnp.allclose(non_normalized_weights_from_fn, non_normalized_weights_from_arr)
    # Without normalization, the weights are unchanged.
    assert jnp.allclose(non_normalized_weights_from_fn, full_expected_weights)

    unit_normalized_weights_from_fn = mu.order_metric_weights(
        rank_to_weight, labels, groups, normalize=mu.Normalization.UNIT_WEIGHTS)
    unit_normalized_weights_from_arr = mu.order_metric_weights(
        weights_arr, labels, groups, normalize=mu.Normalization.UNIT_WEIGHTS)
    average_weight = jnp.average(full_expected_weights)
    assert jnp.allclose(unit_normalized_weights_from_fn, unit_normalized_weights_from_arr)
    assert jnp.allclose(unit_normalized_weights_from_fn, full_expected_weights / average_weight)

    unit_normalized_weights_from_fn = mu.order_metric_weights(
            rank_to_weight, labels, groups, normalize=mu.Normalization.GROUP_UNIT_WEIGHTS)
    unit_normalized_weights_from_arr = mu.order_metric_weights(
        weights_arr, labels, groups, normalize=mu.Normalization.GROUP_UNIT_WEIGHTS)

    # Each group's weight sum should be identical under this normalization.
    assert jnp.allclose(unit_normalized_weights_from_fn, unit_normalized_weights_from_arr)
    norm_weights = unit_normalized_weights_from_fn
    assert (norm_weights[0] + norm_weights[1] ==
            norm_weights[2] + norm_weights[3] ==
            norm_weights[4] + norm_weights[5] + norm_weights[6])
    assert jnp.isclose(jnp.average(norm_weights), 1)
    assert jnp.allclose(unit_normalized_weights_from_fn, unit_normalized_weights_from_arr)

    weights_from_fn = mu.order_metric_weights(
        rank_to_weight, labels, groups, normalize=mu.Normalization.MAX_ONE)
    weights_from_arr = mu.order_metric_weights(
        weights_arr, labels, groups, normalize=mu.Normalization.MAX_ONE)
    assert jnp.allclose(weights_from_fn, weights_from_arr)
    assert jnp.isclose(jnp.average(weights_from_fn * best_order_labels), 1)

    weights_from_fn = mu.order_metric_weights(
        rank_to_weight, labels, groups, normalize=mu.Normalization.MAX_ONE_MIN_ZERO)
    weights_from_arr = mu.order_metric_weights(
        weights_arr, labels, groups, normalize=mu.Normalization.MAX_ONE_MIN_ZERO)
    assert jnp.allclose(weights_from_fn, weights_from_arr)
    assert jnp.isclose(jnp.average(weights_from_fn * best_order_labels), 1)
    assert jnp.isclose(jnp.average(weights_from_fn * worst_order_labels), 0, atol=1e-5)

    weights_from_fn = mu.order_metric_weights(
            rank_to_weight, labels, groups, normalize=mu.Normalization.GROUP_MAX_ONE)
    weights_from_arr = mu.order_metric_weights(
        weights_arr, labels, groups, normalize=mu.Normalization.GROUP_MAX_ONE)
    assert jnp.allclose(weights_from_fn, weights_from_arr)
    assert jnp.isclose(jnp.average(weights_from_fn * best_order_labels), 1)

    weights_from_fn = mu.order_metric_weights(
            rank_to_weight, labels, groups, normalize=mu.Normalization.GROUP_MAX_ONE_MIN_ZERO)
    weights_from_arr = mu.order_metric_weights(
        weights_arr, labels, groups, normalize=mu.Normalization.GROUP_MAX_ONE_MIN_ZERO)
    assert jnp.allclose(weights_from_fn, weights_from_arr)
    assert jnp.isclose(jnp.average(weights_from_fn * best_order_labels), 1)
    assert jnp.isclose(jnp.average(weights_from_fn * worst_order_labels), 0, atol=1e-5)
