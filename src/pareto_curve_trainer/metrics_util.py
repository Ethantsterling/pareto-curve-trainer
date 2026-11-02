"""Utilities for metrics.py."""
from enum import IntEnum
import math
from typing import Callable, NamedTuple
import jax
import jax.numpy as jnp


def log1p_negative_sigmoid(x):
    """A numerically-stable formulation suitable even for extreme xs."""
    # The expressions are mathematically equivalent on real numbers, but the first
    # one is stable for big +x and the second is stable for big -x.
    return jnp.where(x > 0, -x - jnp.log1p(jnp.exp(-x)), -jnp.log1p(jnp.exp(x)))


def streak_lengths(a):
    """Returns the length of each streak in order."""
    streak_ends = 1 + jnp.where(a[1:] != a[:-1])[0]
    streak_ends = jnp.concatenate([jnp.array([0]), streak_ends, jnp.array([len(a)])])
    return streak_ends[1:] - streak_ends[:-1]


def first_index_matching(sorted_array):
    """For each a in sorted_array, find the index of the first i where sorted_array[i] == a."""
    diffs = sorted_array[1:] != sorted_array[:-1]
    diff_mask = jnp.concatenate([jnp.array([True]), diffs])

    indices = jnp.arange(len(sorted_array))
    initial_indices = jnp.where(diff_mask, indices, 0)
    return jax.lax.cummax(initial_indices)


class Normalization(IntEnum):
    """Monotonicity constraints suitable for curves."""
    NONE = 0
    GROUP_MAX_ONE = 1
    GROUP_MAX_ONE_MIN_ZERO = 2
    MAX_ONE = 3
    MAX_ONE_MIN_ZERO = 4


def ideal_and_worst_orders(labels: jnp.ndarray, groups: None | jnp.ndarray=None) -> jnp.ndarray:
    """Returns the indices for the best/worst ordering for labels, optionally within groups."""
    if groups is not None:
        # lexsort is descending, so we invert the labels to sort higher labels to the top.
        ideal_order = jnp.lexsort([-labels, groups])
        worst_order = jnp.lexsort([labels, groups])
    else:
        ideal_order = jnp.argsort(labels, descending=True)
        worst_order = jnp.flip(ideal_order)
    return ideal_order, worst_order


def normalize_weights_to_max_one(
        weights: jnp.ndarray,
        labels: jnp.ndarray,
        groups: None | jnp.ndarray=None) -> jnp.ndarray:
    """Weights that scale a ranking metric to a maximum of 1.
    
    Args:
        weights: The metric's pre-normalization weights. When the metric is evaluated on a score,
            the labels will be sorted by that score. Then the metric output is a weighted averaged
            of the labels (in that order) using the post-normalization weights.
        labels: The float label for each row.
        groups: Optional int group tag for each row, corresponding to the label. Omit for a ranking
            metric that doesn't use groups.
    Returns:
        Post-normalization weights that are a scaling of `weights` such that the metric evaluates
        to 1.0 for the perfect ordering and less than 1.0 for any less-perfect ordering.
    """
    ideal_order, _ = ideal_and_worst_orders(labels, groups)
    max_score = jnp.average(labels[ideal_order] * weights)
    return weights / max_score


def normalize_weights_to_max_one_min_zero(
        weights: jnp.ndarray,
        labels: jnp.ndarray,
        groups: None | jnp.ndarray = None) -> jnp.ndarray:
    """Weights that scale a ranking metric to a minimum of zero and a maximum of 1.

    Args:
        weights: The metric's pre-normalization weights. When the metric is evaluated on a score,
            the labels will be sorted by that score. Then the metric output is a weighted averaged
            of the labels (in that order) using the post-normalization weights.
        labels: The float label for each row.
        groups: Optional int group tag for each row, corresponding to the label. Omit for a ranking
            metric that doesn't use groups.
    Returns:
        Post-normalization weights that are a linear transformation of `weights` such that the
        metric evaluates to 1.0 for the perfect ordering and evaluates to 0.0 for the worst
        possible ordering. Note that it's possible for this normalization to make some weights
        negative.
    """
    ideal_order, worst_order = ideal_and_worst_orders(labels, groups)
    min_score = jnp.average(labels[worst_order] * weights)
    max_score = jnp.average(labels[ideal_order] * weights)

    # Normalize the minimum possible score to 0 and the maximum to 1.
    average_label = jnp.average(labels)
    weights = (weights - min_score / average_label) / (max_score - min_score)
    return weights


def normalized_group_weights_to_max_one(
        weights: jnp.ndarray,
        labels: jnp.ndarray,
        groups:jnp.ndarray) -> jnp.ndarray:
    """Weights that scale a group ranking metric to a maximum of 1.
    
    Args:
        weights: The metric's pre-normalization weights. When the metric is evaluated on a score,
            the labels will be sorted by groups first and that score second. Then the metric output
            is a weighted averaged of the labels (in that order) using the post-normalization
            weights.
        labels: The float label for each row.
        groups: The int group tag for each row, corresponding to the label.
    Returns:
        Post-normalization weights.
    """
    ideal_order, worst_order = ideal_and_worst_orders(labels, groups)
    ideal_order_labels = labels[ideal_order]
    order_groups = groups[ideal_order]  # Depends only on groups, not on labels.
    # For normalization, we need to measure the best and worst scores.
    ideal_order_weighted_labels = weights * ideal_order_labels
    # Normalize each group to a minimum possible score of 0 and a maximum of 1.
    group_sizes = streak_lengths(order_groups)
    # segment_ids is required to index from 0.
    segment_ids = jnp.repeat(jnp.arange(len(group_sizes)), group_sizes)
    group_max_score = jax.ops.segment_sum(ideal_order_weighted_labels, segment_ids) / group_sizes
    # Fill the group scores back out to the entire group.
    group_max_score = jnp.repeat(group_max_score, group_sizes)
    new_weights = weights / group_max_score
    # Skip normalization for groups whose normalization would require dividing by 0.
    # We can't normalize those this way and we shouldn't try.
    return jnp.where(jnp.isfinite(new_weights), new_weights, weights)


def normalized_group_weights_to_max_one_min_zero(
        weights: jnp.ndarray,
        labels: jnp.ndarray,
        groups: jnp.ndarray) -> jnp.ndarray:
    """Weights that scale a group ranking metric to a minimum of zero and a maximum of 1.
    
    Args:
        weights: The metric's pre-normalization weights. When the metric is evaluated on a score,
            the labels will be sorted by groups first and that score second. Then the metric output
            is a weighted averaged of the labels (in that order) using the post-normalization
            weights.
        labels: The float label for each row.
        groups: The int group tag for each row, corresponding to the label.
    Returns:
        Post-normalization weights.
    """
    ideal_order, worst_order = ideal_and_worst_orders(labels, groups)
    ideal_order_labels = labels[ideal_order]
    worst_order_labels = labels[worst_order]
    order_groups = groups[ideal_order]  # Depends only on groups, not on labels.

    # For normalization, we need to measure the best and worst scores.
    ideal_order_weighted_labels = weights * ideal_order_labels
    worst_order_weighted_labels = weights * worst_order_labels

    # Normalize each group to a minimum possible score of 0 and a maximum of 1.
    group_sizes = streak_lengths(order_groups)
    # segment_ids is required to index from 0.
    segment_ids = jnp.repeat(jnp.arange(len(group_sizes)), group_sizes)
    group_max_score = jax.ops.segment_sum(ideal_order_weighted_labels, segment_ids) / group_sizes
    group_min_score = jax.ops.segment_sum(worst_order_weighted_labels, segment_ids) / group_sizes
    group_label_average = jax.ops.segment_sum(ideal_order_labels, segment_ids) / group_sizes
    # Fill the group scores back out to the entire group.
    group_max_score = jnp.repeat(group_max_score, group_sizes)
    group_min_score = jnp.repeat(group_min_score, group_sizes)
    group_label_average = jnp.repeat(group_label_average, group_sizes)
    # I need the average label within each group.
    shifted_weights = (weights - group_min_score / group_label_average)
    normalized_weights = shifted_weights / (group_max_score - group_min_score)

    # Normalization breaks when group_max_score == group_min_score or group_label_average == 0.
    # Those groups can't be normalized to min=0 max=1. It's not clear what to do for them.
    # For now we'll just skip normalizing those groups.
    return jnp.where(jnp.isfinite(normalized_weights), normalized_weights, weights)
