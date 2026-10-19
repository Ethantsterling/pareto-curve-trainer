"""Utilities for examining the Pareto frontier of a collection of points.

Pareto detection: From an array of N-dimensional points, we want to identify:
(1) The points on the pareto frontier, and
(2) The points that are pareto-optimal over their prefix of the points array, whether or not they
are pareto over the entire array.

TODO: add functions to calculate dominated hypervolume.
Dominated hypervolume: To judge a training run and its pareto frontier, we can examine the
volume of the space beneath that pareto frontier (as measured against a baseline).
Calculating N-dimensional hypervolume is infeasible (known algorithms exist, but they're slow and
complex), but Vizier devised an efficient means of approximation in their paper
(https://arxiv.org/pdf/2006.04655.pdf).
It uses random vectors to approximate the dominated hypervolume, relying on a few key observations:
- It's easy to measure the length at which a vector from an origin pierces the pareto frontier.
- Using that length, we can approximate the dominated hypervolume as the volume of a hyperoctant of
a sphere of radius equal to that length.
- The above approximation is inaccurate for one vector, but becomes much more accurate when
averaged over thousands of random vectors.

We can use hypervolume to compare pareto frontiers. Additionally,
we can use the dominated hypervolume of prefixes of the point set to measure the expansion of the
pareto frontier over the course of training. In traditional one-metric training, it's common to
plot the metric against the training step to judge the effectiveness of training. (For example, if
training saturates very early there's probably something wrong, whereas if training never saturates
then we likely need to train longer.) In the multiobjective case, we can plot hypervolume for the
same purpose.
"""
import math
import jax
import jax.numpy as jnp

###########################################################
# CPU-based pareto, suitable for sparse pareto frontiers. #
###########################################################
def _dominated_by(
      points: jnp.ndarray,  # (num_points, num_dimensions)
      p: jnp.ndarray,       # (num_dimension,)
      ) -> jnp.ndarray:
    """Mark points dominated by p."""
    better_on_some_metric = jnp.any(p > points, axis=-1)
    not_worse_on_any_metric = jnp.all(p >= points, axis=-1)
    return better_on_some_metric & not_worse_on_any_metric


# Runs much faster on CPU than on GPU, presumably because CPUs are better at sequential operations
# and conditional branching.
@jax.jit(backend='cpu' )
def pareto(points: jnp.array) -> jnp.array:
    """Returns a boolean mask for the pareto-optimal points.
    
    This algorithm exploits the sparseness of pareto optimal points, so it's efficient in the usual
    case where few points are pareto optimal.
    If instead a large fraction of points are pareto optimal, you're better off using a more
    parallelized algorithm that's tuned for GPU.
    """
    dominated = jnp.zeros(len(points), dtype=bool)

    # For each point that isn't already marked as dominated,
    # mark each point as dominated that that point dominates.
    def _prefix_pareto_one_step(i, val):
        dominated, = val

        new_dominated = jax.lax.cond(
          pred=dominated[i],  # Skip to the next point if the current point is dominated.
          true_fun=lambda: dominated,
          false_fun=lambda: dominated | _dominated_by(points, points[i]),
        )
        return (new_dominated,)

    final_dominated, = jax.lax.fori_loop(
        lower=0,
        upper=len(points), 
        body_fun=_prefix_pareto_one_step,
        init_val=(dominated,),
        )
    return ~final_dominated


def prefix_pareto(points: jnp.ndarray) -> jnp.ndarray:
    """Returns a boolean mask of the points that aren't dominated by any earlier point.
    
    True for points[i] if points[i] is on the pareto frontier of points[:i+1].
    This algorithm exploits the sparseness of pareto optimal points, so it's efficient in the usual
    case where few points are pareto optimal.
    If instead a large fraction of points are pareto optimal, you're better off using a more
    parallelized algorithm that's tuned for GPU.
    """
    step = jnp.arange(len(points)).reshape((-1, 1))
    # Add -step# as an additional metric.
    # That way, a later point can never dominate an earlier point.
    points = jnp.concatenate([points, -step], axis=-1)
    return pareto(points)


##########################################################
# GPU-based pareto, suitable for dense pareto frontiers. #
##########################################################
def _pick_batch_size_for_pareto(points: jnp.ndarray, max_ram=1_000_000_000) -> jnp.ndarray:
    """Pick an appropriate power-of-two batch size for pareto operations.
    
    Fully vectorized pareto calculation costs O(n^2) RAM, which is often too much.
    Going point by point only costs O(n) RAM, but limits our ability to parallelize computation.
    Batching is a compromise that gives us space for parallelization without overflowing RAM.
    """
    points_num_floats = points.shape[0] * points.shape[1]
    points_ram_size = points_num_floats * 4  # float32
    ram_per_map_row = points_ram_size * 3  # Estimate of pareto operation ram.

    max_batch_size = max_ram / ram_per_map_row
    # For a batch size that's a power of 2.
    max_batch_size = 2 ** int(math.log2(max_batch_size))
    max_batch_size = max(max_batch_size, 1)  # Batches can't be smaller than one point!
    batch_size = min(max_batch_size, len(points))
    return batch_size


def _point_first_dominating(
      points: jnp.ndarray,  # (num_points, num_dimensions)
      p: jnp.ndarray,       # (num_dimension,)
      ) -> int:
    """Returns the index of the first point in points that dominates p.

    Returns len(points) if p is pareto-optimal."""
    better_on_some_metric = jnp.any(p > points, axis=-1)
    duplicate = jnp.all(p == points, axis=-1)
    dominates_p = ~(better_on_some_metric | duplicate)

    is_pareto = ~jnp.any(dominates_p)
    return jnp.where(is_pareto, len(points), jnp.argmax(dominates_p))


def _make_first_dominating_point_fn_for_map(points: jnp.ndarray):
    def first_dominating_point(p: jnp.ndarray) -> int:
        return _point_first_dominating(points, p)
    return first_dominating_point


@jax.jit(static_argnames=('batch_size'))
def first_dominating_batched(points: jnp.ndarray, batch_size) -> jnp.ndarray:
    """Output[i] is the index of the first point dominating points[i]."""
    first_dominating_fn = _make_first_dominating_point_fn_for_map(points)
    return jax.lax.map(first_dominating_fn, points, batch_size=batch_size)


def first_dominating(points: jnp.ndarray) -> jnp.ndarray:
    """output[i] is the index of the first point dominating points[i].
    
    If no point dominates points[i], then output[i] is len(points)."""
    batch_size = _pick_batch_size_for_pareto(points)
    return first_dominating_batched(points, batch_size)


def prefix_pareto_dense(points: jnp.ndarray) -> jnp.ndarray:
    """Returns a boolean mask of the points that aren't dominated by any earlier point.
    
    True for points[i] if points[i] is on the pareto frontier of points[:i+1].
    Optimized for GPU. Effective even for dense pareto frontiers.
    """
    return first_dominating(points) <= jnp.arange(len(points))


def _is_point_pareto(
    points: jnp.ndarray,  # (num_points, num_dimensions)
    p: jnp.ndarray,       # (num_dimension,)
    ) -> bool:
    """Check if one point is pareto optimal within an array of points."""
    better_on_some_metric = jnp.any(p > points, axis=-1)
    duplicate = jnp.all(p == points, axis=-1)
    return jnp.all(better_on_some_metric | duplicate)


def _make_point_pareto_fn_for_map(points: jnp.ndarray):
    def point_is_pareto(p: jnp.ndarray) -> int:
        return _is_point_pareto(points, p)
    return point_is_pareto


@jax.jit(static_argnames=('batch_size'))
def _is_pareto_map_batched(points: jnp.ndarray, batch_size) -> jnp.ndarray:
    """Pareto calculation O(n^2) runtime but O(n) memory."""
    p_pareto_fn = _make_point_pareto_fn_for_map(points)
    return jax.lax.map(p_pareto_fn, points, batch_size=batch_size)


def pareto_dense(points: jnp.ndarray) -> jnp.ndarray:
    """Pareto calculation O(n^2) runtime but O(n) memory.
    
    Optimized for GPU. Effective even for dense pareto frontiers.
    """
    # Allow up to 1GB of ram for batching.
    # This is quite conservative but we shouldn't need a huge amount.
    batch_size = _pick_batch_size_for_pareto(points)
    return _is_pareto_map_batched(points, batch_size=batch_size)
