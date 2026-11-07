"""Metrics suitable for training."""
from enum import IntEnum
import math
from typing import Callable, NamedTuple
import jax
import jax.numpy as jnp
import pareto_curve_trainer.metrics_util as metrics_util

#####################
# Pointwise Metrics #
#####################
# The simplest metrics can be evaluated one point at a time, by comparing the prediction to the
# label to produce a point for each score. Then the metric averages those per-point scores to
# produce a final score for the entire dataset.

def mse_fn(label: jnp.ndarray, pred: jnp.ndarray):
    """Vectorized MSE function."""
    # Predictions should be (#trials, #data).
    # labels should be (#data,).
    # Broadcasting works without needing to reshape.
    if jnp.ndim(pred) == 0:
        return (label - pred) ** 2

    if jnp.ndim(pred) == 1:
        # The metric is just one number.
        return -jnp.mean(jnp.square(label - pred))

    # Else, the metric is one number per trial.
    return -jnp.mean(jnp.square(label - pred), axis=-1)


def logloss_fn(label: jnp.ndarray, pred: jnp.ndarray) -> float:
    """Vectorized logloss function.

    Args:
        pred: Log-odds predictions of the label.
        label: The ground truth probability. Booleans, 0-or-1 ints, or floats.
    
    Returns:
        Negated logloss of using pred as the log-odds prediction of label.
    """
    loss = (label * jax.nn.log_sigmoid(pred) +
            (1 - label) * metrics_util.log1p_negative_sigmoid(pred))
    if jnp.ndim(pred) == 0:
        return loss
    return jnp.average(loss, axis=-1)


#################
# Order Metrics #
#################
# For some tasks, we care most about scoring points against each other rather than scoring them
# individually. Order metrics (or ranking metrics) use the model's prediction to order results,
# and then score the resulting order. This approach creates an interesting separation between the
# raw model output and the metric result: if we perform a strictly monotonic transform on the
# raw model output, the transformed version will order results in exactly the same order as the
# raw version, and therefore score exactly the same on order metrics. Consequently:
# - Optimizing order metrics is more flexible than optimizing pointwise metrics.
# - The raw model output has no semantic meaning, because it's only used for order. (Unlike MSE.)
# - To regain semantic meaning, we can map that raw model output to a different distribution.
# Order metrics also aren't differentiable, though there exist differentiable substitutes.
# Most training algorithms need differentiability, but pareto_curve_trainer doesn't, because it
# uses black-box optimization instead of gradients.
# Many order metrics take the form of a weighted sum of the label, with higher weights applied to
# higher-ranking results. This is a flexible format that can handle grouping, normalization, and
# lots of different weighting schemes. To make it effective, we introduce the concept of an
# `anchored metric`: a metric that's been partially precalculated down to a single vector of
# weights, such that we can finish calculating the metric on any particular ordering by multiplying
# the weights vector against the ordered labels vector and averaging the results.
# We call this metric 'anchored' because it's locked to its data set: its weights are determined
# from labels (and groups for a group metric) from the data, so the resulting anchored metric can
# only be used to evaluate reorderings of the rows of that one data set.
# This is optimized for the standard training use case: recalculating the same metric over and
# over on the same dataset with different orderings.
# Also, the tasks that are difficult to do in jitted JAX (such as averaging within a group and then
# averaging across groups) can be precalculated so that we perform them once in non-jitted code and
# then never need to do them in our tight jitted loop.
def _eval_order_metric(weight_by_position: jnp.ndarray,
                       label_by_position: jnp.ndarray) -> float:
    """Evaluate an order metric on ranks.
    
    Args:
        `weight_by_position`: 1D array that characterizes an order metric on a particular dataset.
            Independent of any particular scoring/ranking model. If the order metric uses groups
            or normalization, then weight_by_position must already incorporate the consequence of
            those groupings or that normalization.
        `labels_by_position`: 1D array of the labels as ordered by some scoring/ranking model.
    
    Returns:
        Float of the average of weight * label. Note that this is not a weighted average!
    """
    return jnp.average(weight_by_position * label_by_position)


class AnchoredOrderMetric(NamedTuple):
    """An order metric defined on pre-specified data.

    When an order metric is applied repeatedly to the same dataset, we can preprocess the data to
    calculate the position weights implied by the metric, label, normalization, groupings, etc.
    This speeds up and simplifies evaluation, since we only need to calculate those weights once.
    However, an AnchoredOrderMetric is only valid on the same labels & groups used to define it.

    By wrapping this metric in a pytree (as opposed to a function with a closure), we ensure good
    behavior if the metric is passed from non-jitted code into jitted code. The weights will be
    treated as data rather than static/constant. (JAX's jitting needs large arrays to be passed
    as data rather than as constants, because it'll unroll them if they're passed as constants.)
    """
    weight_by_position: jnp.ndarray  # 1D float array

    def eval(self, label_by_position: jnp.ndarray) -> float:
        """Evaluate an order metric on ranks.
        
        Args:
            `labels_by_position`: 1D array of the labels as ordered by some scoring/ranking model.
            Higher model scores get ranked to lower position indices, so that rank=0 is the highest
            rank. (If the metric uses groups, then rank primarily by group and secondarily by model
            score.)
        
        Returns:
            Float of the weighted average of the label.
        """
        return _eval_order_metric(label_by_position, self.weight_by_position)


def anchor_order_metric(rank_to_weight: jnp.ndarray | Callable,
                        labels: jnp.ndarray,
                        groups: None | jnp.ndarray=None,
                        normalize=metrics_util.Normalization.GROUP_UNIT_WEIGHTS,
                        ) -> AnchoredOrderMetric:
    """Affix an order metric to data.

    The resulting AnchoredOrderMetric can be evaluated on any valid reranking of the labels.
    (A valid reranking must respect the groups if groups are provided.)
    However, the AnchoredOrderMetric is only valid on the same labels & groups used to define it.

    This function isn't jittable in JAX, but its output is jittable.
    
    Args:
        rank_to_weight: Maps a row's position index (after sorting) to its corresponding weight.
            If groups are present, then the position index is measured within its group.
            `rank_to_weight` can either be a function or an array. If a function, it must map
            every position index to a weight float. If an array, any position index that
            exceeds the array's bounds will be mapped to a weight of 0. (This is common behavior
            for @K metrics, which only care about the K top-ranked rows.)
            This should normally be a monotonically decreasing function, such that a
            higher position index (i.e. a worse rank) always corresponds to a lower weight.
        labels: 1D floats. Ground truth for each row of data.
        groups: Optional 1D ints, used for group metrics. Rows i and j are in the same group
            iff groups[i] == groups[j]. `groups` must be ordered to match labels, so that
            groups[i] is the group of labels[i].
        normalize: An IntEnum indicating what normalization (if any) to apply. By default, we
            normalize the weights to an average of 1 (or a per-group average of 1 if using
            groups).
    
    Returns:
        An AnchoredOrderMetric prepared for evaluation on the labels sorted primarily by the groups
        (if groups exist) and secondarily by a model's score. This AnchoredOrderMetric incorporates
        the requested normalization, allowing for metrics like nDCG.
    """
    anchored_weights = metrics_util.order_metric_weights(rank_to_weight, labels, groups, normalize)
    return AnchoredOrderMetric(anchored_weights)

#######################
# Group Order Metrics #
#######################
# Metrics such as NDCG require grouped data, where ranking happens only between elements in the
# same group. If groups were all the same size, we could just use a dimension to indicate group.
# Unfortunately, many ranking tasks have groups of differing size, so we'll take group data as a
# separate array of group IDs corresponding to each row.
# Normalization also becomes more involved, because some common group order metrics (e.g. nDCG)
# normalize within groups.
def anchored_ndcg_metric(
        labels: jnp.ndarray,
        groups: jnp.ndarray,
        depth: None | int=None) -> AnchoredOrderMetric:
    """Normalized Distributed Cumulative Gain metric."""
    def rank_to_weight(rank):
        if depth is None:
            return 1 / jnp.log2(rank + 2)
        return jnp.where(rank < depth, 1 / jnp.log2(rank + 2), 0)

    return anchor_order_metric(
        rank_to_weight,
        labels,
        groups,
        normalize=metrics_util.Normalization.GROUP_MAX_ONE)


def group_rank(predictions, groups):
    """Rank primarily by groups (ascending) and secondary by predictions (descending).
    
    Alternative to jnp.lexsort that's ~3x faster on GPU for ~1M points.
    Ranking metrics require this kind of sorting, which is a bottleneck for training.
    """
    pred_order = jnp.argsort(predictions, descending=True)
    group_pred_order = jnp.argsort(groups[pred_order])
    return pred_order[group_pred_order]