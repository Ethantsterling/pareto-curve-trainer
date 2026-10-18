"""Multiobjective pareto selection logic for identifying the best N members of a population.

Adapted from Vizier's NSGA2 survival logic
(https://github.com/google/vizier/blob/main/vizier/_src/algorithms/evolution/nsga2.py)
with a few major differences:
*  This code is written in jittable JAX for performance.
*  This code generalizes Vizier's concept of contraint_violation by allowing the user to specify
   an optional priority score that takes precedence over the other scores. (This is equivalent to
   Vizier's constraint_violation if the priority score equals the number of constraints violated.)
   The other scores are still used for tiebreaking when population members have equal priority
   scores. The priority score is expected to have many ties -- if it didn't, that would defeat
   the purpose of using multiple objectives!

As in Vizier's logic, we resolve conflicts between multiple metrics by selecting our subpopulation
to be as close to pareto optimal as we can, and use the crowding distance between points as a last
tiebreaker to spread the chosen points out as much as we can over the frontier. (Without crowding
distance, we'd risk gathering a cluster of pareto-optimal points in one corner of the frontier.)
"""
from enum import IntEnum
from typing import Sequence, Tuple
import jax
import jax.numpy as jnp


def simple_cull_indices(scores, target_size) -> jax.Array:
    """Single-metric culling. Returns a boolean mask of the target_size best scores."""
    # Consider using an argpartition instead of argsort.
    top_indices = jnp.argsort(scores, descending=True)[:target_size]
    # To avoid re-ordering the population, we sort the kept indices.
    return jnp.sort(top_indices)


def pareto_domination(ys: jnp.ndarray) -> jnp.ndarray:
    """Measures whether ys[i] dominates ys[j] for each i,j.

    Args:
      ys: (number of population) x (number of metrics) array.

    Returns:
      2D bool array where [i][j] == True iff ys[i] dominates ys[j].
    """
    # Reshape to broadcast against ys.
    ys_prime = ys.reshape([ys.shape[0], 1, ys.shape[-1]])
    # (p, p, m) matrices where [i][j] compares ys[i] to ys[j]
    ys_prime_gt = ys_prime > ys
    ys_prime_gte = ys_prime >= ys
    # Collapse the metric dimension, reducing to (p, p).
    # dominated[i][j] means ys[i] dominates ys[j].
    dominated = jnp.all(ys_prime_gte, axis=-1) & jnp.any(ys_prime_gt, axis=-1)
    return dominated


@jax.jit
def pareto_rank(ys: jnp.ndarray) -> jnp.ndarray:
    """Pareto rank, which is the number of points dominating it.

    Args:
      ys: (number of population) x (number of metrics) array.

    Returns:
      (number of population,) integer array.
    """
    if ys.shape[0] == 0:
        return jnp.zeros([0])
    dominated = pareto_domination(ys)
    return jnp.sum(dominated, axis=0)


@jax.jit
def pareto_rank_with_exclusion(ys: jnp.ndarray, exclusion_mask) -> jnp.ndarray:
    """The number of dominating points, not counting those excluded.

    Args:
      ys: (number of population) x (number of metrics) array.

    Returns:
      (number of population,) integer array.
    """
    if ys.shape[0] == 0:
        return jnp.zeros([0])
    dominated = pareto_domination(ys)
    empty_row = jnp.zeros(ys.shape[0], dtype=bool)
    # Overwrite domination so those excluded don't count.
    dominated = jnp.where(exclusion_mask.reshape((-1, 1)), dominated, empty_row)
    return jnp.sum(dominated, axis=0)


@jax.jit
def _min_masked(ys, mask):
    """Replace ys's False-masked rows with the minimum non-masked column values."""
    # Find the minimum of each column, not counting masked rows.
    mask = mask.reshape(-1, 1)
    ys_with_rows_masked = jnp.where(mask, ys, jnp.nan)
    col_mins = jnp.nanmin(ys_with_rows_masked, axis=0)
    # Overwrite masked rows with the row of minimum column values.
    return jnp.where(mask, ys, col_mins)


@jax.jit
def crowding_distance(
    ys: jnp.ndarray,
    *,
    exclusion_mask: jnp.ndarray | None = None,
    tiebreaker_random_key: jnp.ndarray | None = jax.random.key(12345),
    extra_tiebreakers: Sequence[jnp.ndarray] = tuple()) -> jnp.ndarray:
    """Crowding distance to reward points that are further awy from others.

    Adapted from Vizier's NSGA2 crowding distance, which references
    https://medium.com/@rossleecooloh/optimization-algorithm-nsga-ii-and-python-package-deap-fca0be6b2ffc
    This version is written in jittable JAX, which requires us to know array size ahead of time.
    We get around that limitation with the `exclusion_mask` parameter, which indicates which points
    we're actually using to compute crowding distance.

    Args:
      ys: (size of population) x (number of metrics) array.
      exclusion_mask: (size of population,) array of bools. When calculating
        the crowding distance, we ignore the False values.
      extra_tiebreakers: A sequence of (size of population,) arrays of floating
        numbers. If specified, they are used to break ties when sorting the
        population. By default, a random number is always used to break ties
        consistently across all metrics.

    Returns:
      (size of population,) float32 array. Higher numbers mean less crowding
      and more desirable.
    """
    scores = jnp.zeros([ys.shape[0]], dtype=jnp.float32)

    if ys.shape[0] <= 1:
        return scores
    if exclusion_mask is None:
        exclusion_mask = jnp.ones(ys.shape[0], dtype=bool)

    # set ys[exclusion_mask] to the minimum non-masked values of each column,
    # so that they don't interfere with the crowding score calculation for the
    # other rows.
    ys = _min_masked(ys, mask=exclusion_mask)

    # Use a random number to break ties, with the same number for each metric.
    tiebreakers = list(extra_tiebreakers)
    if tiebreaker_random_key is not None:
        random_tiebreaker = jax.random.normal(key=tiebreaker_random_key, shape=ys.shape[0])
        tiebreakers += [random_tiebreaker]

    # m-th metric.
    for m in range(ys.shape[1]):
        yy = ys[:, m]
        # lexsort keys are ordered from least significant to most significant,
        # and all sorting is in ascending order.
        # The exclusion_mask forces excluded elements to the lowest index of `sid`,
        # because False precedes True in ascending order.
        # `yy` sorts in ascending order of the mth metric.
        # tiebreakers break ties.
        sid = jnp.lexsort(tiebreakers[::-1] + [yy] + [exclusion_mask])

        # Compute the range of the m-th metric.
        # Excluded elements are all overwritten with the minimum metrics, so sid[0]
        # always equals the minimum of the non-excluded elements
        yrange = yy[sid[-1]] - yy[sid[0]] + jnp.finfo(jnp.float32).eps

        # Lower boundary receives a one-sided score.
        at0_increase = (yy[sid[1]] - yy[sid[0]]) / yrange
        scores = scores.at[sid[0]].add(at0_increase)
        # Upper boundary is assigned infinity.
        scores = scores.at[sid[-1]].add(jnp.inf)

        mid_score = (yy[sid[2:]] - yy[sid[:-2]]) / yrange
        scores = scores.at[sid[1:-1]].add(mid_score)
    # Normalize the score to [0, 1].
    return scores / ys.shape[1]


class Select(IntEnum):
    """Tracking flag for which points to discard/consider/keep."""
    EXCLUDE = 0
    BORDER = 1
    INCLUDE = 2


def select_by(
    ys: jnp.ndarray,
    target: int,
    tiebreaker: None | jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns a trinary mask identifying the top `target` elements of ys.
    
    Bins inputs into the three categories of a Select enum (EXCLUDE/BORDER/INCLUDE).
    We rank by ys (higher is better) and break ties by tiebreaker.
    INCLUDE indicates entries that unambiguously rank in the top target entires.
    BORDER indicates tied entries that could rank in the top target but can't all fit.
    EXCLUDE indicates entries that definitely do not rank in the top target.

    Args:
      ys: Array of shape (#points, #metrics) that order selection. Higher is better.
      target: Count to tag as INCLUDE, unless there's a BORDER.
      tiebreaker: Array of shape (#points,) that breaks ties in selection when
        ys[i] == ys[j]. Higher is better.
  
    Returns:
      An output array of shape (#metrics,) of Select int enums.
      * `output == Select.INCLUDE` for <= `target` values, indicating they're
        definitely selected.
      * `output == Select.INCLUDE | output == Select.BORDER` for >= `target` values.
      * `output == Select.BORDER` are identical in both ys and tiebreaker to the
        target-th best entry, indicating a tie between at least two entries.
        If the caller needs precisely `target` top values, the caller must break
        the tie subsequently.
      * `output == Select.EXCLUDE` should be discarded.
    """
    if len(ys) <= target:
        return jnp.repeat(Select.INCLUDE, len(ys))
    if tiebreaker is None:
        tiebreaker = jnp.ones(len(ys))

    # ys take precedence over tiebreaker.
    sort_index = jnp.lexsort([tiebreaker, ys])
    cutoffindex = sort_index[-target]
    cutoffnumber = ys[cutoffindex]
    cutofftiebreaker = tiebreaker[cutoffindex]

    # Bool arrays.
    include = ((ys > cutoffnumber) |
              ((ys == cutoffnumber) & (tiebreaker > cutofftiebreaker)))
    border = (ys == cutoffnumber) & (tiebreaker == cutofftiebreaker)
    # If the 'border' actually has a perfect cutoff, then actually the entire
    # 'border' should just be INCLUDE and there should be no BORDER.
    include_whole_border = jnp.sum(include | border) <= target

    # Includes are always tagged as 2.
    included_tag = 2 * include
    # The border is tagged as 1 unless we include_whole_border, then they're 2.
    border_tag = border * (include_whole_border + 1)
    return included_tag + border_tag


# @jax.jit(static_argnames='target_size')
def select(scores: jnp.ndarray,   # 2D array of metric x point
           target_size: int,      # Number of indices to return
           priority_score: jnp.ndarray | None=None,
           )-> jnp.ndarray:
    """Applies survival process.

    Sorted all points by 3-tuple
    1. Descending order of priority_score, if provided.
    2. Ascending order of how many points it's dominated by in the other scores.
    3. Descending order of crowding distance.
    Each of 1,2,3 depends on the previous ones, so we have to generate them in
    order and pass through the information about which ones are kept/dropped.

    To be JAX-jittable, each array's size has to be determined at runtime.
    Unlike Vizier's version, which relies on numpy, we can't actually filter the
    selection arrays down to smaller arrays. Instead, we track the selection as a
    trinary array of INCLUDE/BORDER/EXCLUDE, using BORDER for the poitns that are
    yet to be decided.

    Args:
      scores: 2D array of metric x point. Higher values are always better.
      target_size: The number of points to keep.
      priority_score: Optional 1D array with one score for each point. If present,
          `priority_score` takes precedence. Higher values are always better.
          Note that `scores` only matter when `priority_score` is tied for two
          points. `priority_score` is a generalization of Vizier NSGA2's
          contraint metrics.

    Returns:
      Array of shape (target_size,) of indices for the selected rows.
    """
    all_indices = jnp.arange(len(scores))
    if len(scores) <= target_size:
        return all_indices

    # Sort by the safety constraint.
    if priority_score is None:
        # Everything is on the 'border'.
        selection = jnp.ones(len(scores))
    else:
        selection = select_by(priority_score, target=target_size)

    # Calculate each point's pareto domination (the number of other points that
    # dominate it). Ignore domination by rows we've already
    # marked as EXCLUDE in the selection.
    # NOTE: this differs from visier NSGA2, which ignores domination both by
    # INCLUDE and EXCLUDE rows. In my opinion, it makes sense to count domination
    # by INCLUDE rows.
    pareto_ranks = pareto_rank_with_exclusion(
        scores, exclusion_mask=selection != Select.EXCLUDE)

    # Select all INCLUDE and select the least-dominated points from BORDER.
    # Use -pareto_ranks so that less-dominated points rank higher.
    selection = select_by(
        selection, target=target_size, tiebreaker=-pareto_ranks
    )

    # Measure crowding distance to reward points that are more spread out on the
    # pareto frontier. A higher distance is better.
    # Ignore crowding from points where the selection is EXCLUDE.
    distance = crowding_distance(
        scores,
        exclusion_mask=selection != Select.EXCLUDE,
        extra_tiebreakers=[pareto_ranks],
    )
    selection = select_by(
        selection, target=target_size, tiebreaker=distance,
    )
    # The crowding_distance should settle the border, so selection should now
    # consist exclusively of 2's and 0's.
    # We want to return the indices of the 2's, of which there should be precisely
    # `target_size`.

    # To go from selection tags to indices, we need to do... an argpartition?
    # By construction there should be exactly `target_size` 2's in `selection`.
    # But JAX can't infer that.
    return jnp.argpartition(selection, kth=-target_size)[-target_size:]
