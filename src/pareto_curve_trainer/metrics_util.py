"""Utilities for metrics.py."""

from collections.abc import Callable
from enum import IntEnum

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
    UNIT_WEIGHTS = 1  # Weights average to 1.
    GROUP_UNIT_WEIGHTS = 2  # Weights average to 1 in each group.
    MAX_ONE = 3  # Highest possible score is 1.
    MAX_ONE_MIN_ZERO = 4  # Lowest possible score is 0, highest is 1.
    GROUP_MAX_ONE = 5  # Highest possible score in each group is 1.
    GROUP_MAX_ONE_MIN_ZERO = 6  # Lowest possible score in each group is 0, highest is 1.


def ideal_and_worst_orders(labels: jnp.ndarray, groups: None | jnp.ndarray = None) -> jnp.ndarray:
    """Returns the indices for the best/worst ordering for labels, optionally within groups."""
    if groups is not None:
        # lexsort is descending, so we invert the labels to sort higher labels to the top.
        ideal_order = jnp.lexsort([-labels, groups])
        worst_order = jnp.lexsort([labels, groups])
    else:
        ideal_order = jnp.argsort(labels, descending=True)
        worst_order = jnp.flip(ideal_order)
    return ideal_order, worst_order


def normalize_weights_to_unit(weights: jnp.ndarray) -> jnp.ndarray:
    """Weights that average to 1."""
    return weights / jnp.average(weights)


def normalize_group_weights_to_unit(weights: jnp.ndarray, groups: jnp.ndarray) -> jnp.ndarray:
    """Weights that average to 1 and also equalize each group's total weight."""
    groups = groups.sort()
    group_sizes = streak_lengths(groups)

    # segment_ids is required to index from 0.
    segment_ids = jnp.repeat(jnp.arange(len(group_sizes)), group_sizes)
    per_group_average_weight = jax.ops.segment_sum(weights, segment_ids)
    per_group_average_weight = jnp.repeat(per_group_average_weight, group_sizes)
    group_unit_weights = weights / per_group_average_weight / len(group_sizes) * len(groups)
    # No group should have zero average weights, but if for some reason they do, normalization
    # breaks and we fall back to leaving the weights unchanged.
    return jnp.where(jnp.isfinite(group_unit_weights), group_unit_weights, weights)


def normalize_weights_to_max_one(
    weights: jnp.ndarray, labels: jnp.ndarray, groups: None | jnp.ndarray = None
) -> jnp.ndarray:
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
    weights: jnp.ndarray, labels: jnp.ndarray, groups: None | jnp.ndarray = None
) -> jnp.ndarray:
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


def normalize_group_weights_to_max_one(
    weights: jnp.ndarray, labels: jnp.ndarray, groups: jnp.ndarray
) -> jnp.ndarray:
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
    ideal_order, _ = ideal_and_worst_orders(labels, groups)
    ideal_order_labels = labels[ideal_order]
    order_groups = groups[ideal_order]  # Depends only on groups, not on labels.
    # For normalization, we need to measure the best and worst scores.
    ideal_order_weighted_labels = weights * ideal_order_labels
    # Normalize each group to a minimum possible score of 0 and a maximum of 1.
    group_sizes = streak_lengths(order_groups)
    # segment_ids is required to index from 0.
    segment_ids = jnp.repeat(jnp.arange(len(group_sizes)), group_sizes)
    group_max_score = jax.ops.segment_sum(ideal_order_weighted_labels, segment_ids)
    # Fill the group scores back out to the entire group.
    group_max_score = jnp.repeat(group_max_score, group_sizes)
    new_weights = weights * len(groups) / len(group_sizes)
    normalized_weights = new_weights / group_max_score
    # Skip normalization for groups whose normalization would require dividing by 0.
    # We can't normalize those this way and we shouldn't try.
    return jnp.where(jnp.isfinite(normalized_weights), normalized_weights, weights)


def normalize_group_weights_to_max_one_min_zero(
    weights: jnp.ndarray, labels: jnp.ndarray, groups: jnp.ndarray
) -> jnp.ndarray:
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
    group_max_score = jax.ops.segment_sum(ideal_order_weighted_labels, segment_ids)
    group_min_score = jax.ops.segment_sum(worst_order_weighted_labels, segment_ids)
    group_label_average = jax.ops.segment_sum(ideal_order_labels, segment_ids)
    # Fill the group scores back out to the entire group.
    group_max_score = jnp.repeat(group_max_score, group_sizes)
    group_min_score = jnp.repeat(group_min_score, group_sizes)
    group_label_average = jnp.repeat(group_label_average, group_sizes)
    # I need the average label within each group.
    shifted_weights = weights - group_min_score / group_label_average
    normalized_weights = shifted_weights / (group_max_score - group_min_score)
    normalized_weights = normalized_weights * len(groups) / len(group_sizes)

    # Normalization breaks when group_max_score == group_min_score or group_label_average == 0.
    # Those groups can't be normalized to min=0 max=1. It's not clear what to do for them.
    # For now we'll just skip normalizing those groups.
    return jnp.where(jnp.isfinite(normalized_weights), normalized_weights, weights)


def within_group_rank(groups: jnp.ndarray) -> jnp.ndarray:
    """The rank of each row within its group after sorting by groups."""
    group_first_index = first_index_matching(groups.sort())
    absolute_index = jnp.arange(len(groups))
    return absolute_index - group_first_index


def order_metric_weights(
    rank_to_weight: jnp.ndarray | Callable,
    labels: jnp.ndarray,
    groups: None | jnp.ndarray = None,
    normalize: Normalization = Normalization.GROUP_UNIT_WEIGHTS,
) -> jnp.ndarray:
    """Calculate the weights for an order metric.

    Args:
        rank_to_weight: Maps a row's position (after sorting) to its corresponding weight.
            If groups are present, then the position is measured within its group.
            `rank_to_weight` can either be a function or an array. If a function, it must map
            every position index to a weight float. If an array, any position index that
            exceeds the array's bounds will be mapped to a weight of 0.
            This should normally be a monotonically decreasing function, such that a
            higher rank index (i.e. a worse rank) always corresponds to a lower weight.
        labels: 1D floats. Ground truth for each row of data.
        groups: Optional 1D ints, used for group metrics. Rows i and j are in the same group
            iff groups[i] == groups[j]. `groups` must be ordered to match labels, so that
            groups[i] is the group of labels[i].
        normalize: An IntEnum indicating what normalization (if any) to apply. By default, we
            normalize the weights to an average of 1 (or a per-group average of 1 if using
            groups).

    Returns:
        The final weights array after (optional) grouping and (optional) normalization, whose
        length equals the length of `labels`.
    """
    if not isinstance(rank_to_weight, Callable):
        # Map anything with rank >= len(rank_to_weight) to a weight of 0.
        rank_to_weight_array = jnp.concatenate([rank_to_weight, jnp.array([0])])
        rank_to_weight = lambda indices: rank_to_weight_array[indices]

    if groups is None:
        # If no groups are specified, that should be equivalent to putting everything in one group.
        if normalize == Normalization.GROUP_UNIT_WEIGHTS:
            normalize = Normalization.UNIT_WEIGHTS
        if normalize == Normalization.GROUP_MAX_ONE:
            normalize = Normalization.MAX_ONE
        if normalize == Normalization.GROUP_MAX_ONE_MIN_ZERO:
            normalize = Normalization.MAX_ONE_MIN_ZERO

    ranks = jnp.arange(len(labels)) if groups is None else within_group_rank(groups)
    weights = rank_to_weight(ranks)
    if normalize == Normalization.UNIT_WEIGHTS:
        return normalize_weights_to_unit(weights)
    if normalize == Normalization.GROUP_UNIT_WEIGHTS:
        return normalize_group_weights_to_unit(weights, groups)
    if normalize == Normalization.MAX_ONE:
        return normalize_weights_to_max_one(weights, labels, groups)
    if normalize == Normalization.MAX_ONE_MIN_ZERO:
        return normalize_weights_to_max_one_min_zero(weights, labels, groups)
    if normalize == Normalization.GROUP_MAX_ONE:
        return normalize_group_weights_to_max_one(weights, labels, groups)
    if normalize == Normalization.GROUP_MAX_ONE_MIN_ZERO:
        return normalize_group_weights_to_max_one_min_zero(weights, labels, groups)
    return weights
