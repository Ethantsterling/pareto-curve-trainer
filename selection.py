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
      (number of population) integer array.
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
      (number of population) integer array.
    """
    if ys.shape[0] == 0:
        return jnp.zeros([0])
    dominated = pareto_domination(ys)
    empty_row = jnp.zeros(ys.shape[0], dtype=bool)
    dominated = jnp.where(exclusion_mask.reshape((-1, 1)), dominated, empty_row)
    # Overwrite domination so those excluded don't count.
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
    """Crowding distance.

    Reference:
    https://medium.com/@rossleecooloh/optimization-algorithm-nsga-ii-and-python-package-deap-fca0be6b2ffc
    except that the lower boundary does not get infinity crowding score.

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
    # ys = jnp.where(exclusion_mask.reshape((-1, 1)), ys, ys.min(axis=0))
    ys = _min_masked(ys, mask=exclusion_mask)

    # Use a random number to break ties. But the same random number is used for
    # all metrics.
    tiebreakers = list(extra_tiebreakers)
    if tiebreaker_random_key is not None:
        random_tiebreaker = jax.random.normal(key=tiebreaker_random_key, shape=ys.shape[0])
        tiebreakers += [random_tiebreaker]
      # Random tiebreaker interferes with the masking.
      # tiebreakers = list(extra_tiebreakers) + [random_tiebreaker]

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

        # Lower boundary is assigned a one-sided score and does not automatically
        # get infinity. This is different from the paper. The lower boundary means
        # it's dominated by all other points in one dimension. There's no reason to
        # favor it over other points.
        at0_increase = (yy[sid[1]] - yy[sid[0]]) / yrange
        scores = scores.at[sid[0]].add(at0_increase)
        # Upper boundary is assigned infinity. This point will survive anyways
        # because it's pareto-optimal. But in case there are ties, it's useful to
        # make only one of them stand out.
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


# Derivative of Vizier's nsga2, which uses numpy rather than JAX.
## To allow sequential selections, maybe _select_by should receive a selection
## array and then only perform selection logic on the border?
## That way we can use the true ys and the true target to each selection.
## Note that if we pass the true target (instead of the remaining target) then
## we can afford to set target as dynamic, because it will rarely change.
## I believe that allows for the simpler O(N) partition-based solution.
## How to enforce inputted INCLUDE/EXCLUDE? Either stick selections and ys
## together into a 2D array and do partitioning on the 2D row, sorting first by
## selection and second by y, or else override ys by setting ys with EXCLUDE
## to -inf and ys with INCLUDE to +inf. That assumes that the real ys are never
## -inf or +inf, which should be true but isn't guaranteed, so I'd prefer not to
## rely on it.
# @jax.jit
def _select_by(ys: jnp.ndarray,
               target: int,
               preselection: None | jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns a boolean index array for the top `target` elements of ys.

    Args:
      ys: Array of shape [M]. Entries are expected to have a small set of unique
        values.
      target: Count to return.

    Returns:
      An output array of Select int enums
      * `ys[output == Select.INCLUDE]` has length less than or equal to `target`.
          They are within top `target`.
      * `ys[output == Select.INCLUDE | output == Select.BORDER]` has length
          greater than or equal to `target`.
      * `ys[output == Select.BORDER]` have all-identical entries. Callers should
          break ties among them.
      * `ys[output == Select.EXCLUDE]` should be discarded.
    """
    # if len(ys) <= target:
    #   return jnp.repeat(Select.INCLUDE, len(ys))
    if preselection is None:
        preselection = jnp.ones(len(ys))

    ## O(nlogn) solution.
    unique, counts = jnp.unique(ys, return_counts=True, size=len(ys) )
    unique, counts = unique[::-1], counts[::-1]  # We want descending order.
    counts_cumsum = jnp.cumsum(counts)
    cutoffidx = jnp.argmax(counts_cumsum >= target)
    cutoffnumber = unique[cutoffidx]
    # We have a border separate from includes iff we didn't overshoot `target`.
    include_whole_border = counts_cumsum[cutoffidx] <= target

    # counts_cumsum[cutoffidx] == target --> everything is included, no border.
    # counts_cumsum[cutoffidx] > target --> we have a border.

    ## Problem with using partitioned: we'd have to re-jit for different targets.
    ## That's... probably fine, though? It's not as if we're going to be constantly
    ## changing the target number.
    ## O(n) solution, requires re-jitting.
    # partitioned = jnp.partition(ys, -target)
    # cutoffnumber = partitioned[-target]
    # include_whole_border = jnp.sum(ys >= cutoffnumber) <= target


    include = (ys > cutoffnumber).astype(int)
    border = (ys == cutoffnumber).astype(int)
    # Includes are always tagged as 2.
    included_tag = 2 * include
    # The border is tagged as 1 unless we include_whole_border, then they're 2.
    border_tag = border * (include_whole_border + 1)
    return included_tag + border_tag


# We'll have to make target dynamic, but that's fine.
# It just means rejitting every time the target size changes.
def _select_by_with_tiebreaker(
    ys: jnp.ndarray,
    target: int,
    tiebreaker: None | jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns a boolean index array for the top `target` elements of ys.

    Args:
      ys: Array of shape [M] that order selection. Higher is better.
      target: Count to tag as INCLUDE, unless there's a BORDER.
      tiebreaker: Array of shape [M] that breaks ties in selection when
        ys[i] == ys[j]. Higher is better.
      Entries with a higher value take precedence
        over entries with a lower value. When there's a tie, ys prevail.. Entries are 0/1/2 Select enums, indicating
        inputs in y that are locked to EXCLUDE, are on the BORDER, or locked to
        EXCLUDE. Our output should always favor INCLUDE > BORDER > EXCLUDE.


    Returns:
      An output array of shape [M] of Select int enums.
      * `output == Select.INCLUDE` for <= `target` values, indicating they're
        definitely selected.
      * `output == Select.INCLUDE | output == Select.BORDER` for >= `target` values.
      * `output == Select.BORDER` are identical in both ys and tiebreaker to the
        target-th best entry, indicating a tie between at least two entries.
        If the caller needs precisely `target` top values, the caller must break
        the tie in
      * `output == Select.EXCLUDE` should be discarded. They
    """
    # print(f'_select_by {target} on {ys} with tiebreaker {tiebreaker}')
    if len(ys) <= target:
        return jnp.repeat(Select.INCLUDE, len(ys))
    if tiebreaker is None:
        tiebreaker = jnp.ones(len(ys))

    ## Is there a way to rewrite this to use an arbitrary sequence of deciders?
    ## That would feel more general.
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
        selection = _select_by(priority_score, target=target_size)

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
    selection = _select_by_with_tiebreaker(
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
    selection = _select_by_with_tiebreaker(
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