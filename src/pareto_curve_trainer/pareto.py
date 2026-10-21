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
# jax.jit(backend=...) is deprecated, so instead we use a non-jitted wrapper function that puts the
# data onto the CPU before calling this function.
@jax.jit
def _pareto(points: jnp.array) -> jnp.array:
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


def pareto(points: jnp.array) -> jnp.array:
    """Returns a boolean mask for the pareto-optimal points.
    
    A CPU algorithm that exploits the sparseness of pareto optimal points. Efficient in the usual
    case where few points are pareto optimal.
    If instead a large fraction of points are pareto optimal, you're better off using a more
    parallelized algorithm that's tuned for GPU.
    """
    cpu = jax.devices('cpu')[0]
    points = jax.device_put(points, cpu)
    return _pareto(points)


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
    return first_dominating(points) >= jnp.arange(len(points))


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


####################################################
# Prefix dominated hypervolume. Suitable for GPU.  #
####################################################
def _normalize_vectors(vectors):
    """Normalize the vectors to length 1."""
    return vectors / jnp.linalg.norm(vectors, axis=-1).reshape((-1, 1))


def _normalize_scores(scores):
    """Normalize the scores so that shorter scores don't hog the vectors."""
    means = jnp.mean(scores, axis=0)
    # This is a simple normalization by the arithmetic mean.
    # Maybe we should weight the mean? Weight each score by the product of its other coordinates?
    # Anyway, regardless of how we normalize, as long as we save the multipliers
    # and return them, we can reconstruct the proper hypervolume.
    return scores / means, jnp.prod(means)


def _vector_depth(points: jnp.ndarray, vector: jnp.ndarray):
    """Measure how deeply the vector pierces the dominated hypervolume."""
    ratios = points / vector  # 2D
    depth = jnp.min(ratios, axis=-1)  # 1D
    return jax.lax.cummax(depth, axis=0)  # 1D


def unit_hyperoctant_volume(dimension):
    """Hypervolume of a hyperoctant of a hypersphere with radius 1.

    This is a multiplier that applies to our dominated hypervolume to make it
    accurate. This is unlikely to matter since we normally use hypervolume as a
    relative measure rather than caring about its absolute meaning, but by
    including this term we make it robust just in case.
    """
    return (math.pi**(dimension / 2) /
            math.gamma(dimension / 2 + 1) /
            2**dimension)

def _prefix_dominated_hypervolume(points, vectors):
    """Assumes an origin at zero, and few enough vectors to fit intermediates in RAM."""
    # depths is (#points, #vectors).
    depths = jax.vmap(_vector_depth, in_axes=(None, 0))(points, vectors)
    # Treat each vector's depth as the radius of a hyperoctant.
    # The hyperoctant is an estimate of the total hypervolume.
    dimension = points.shape[-1]
    hyper_octant_volume = depths ** dimension * unit_hyperoctant_volume(dimension)
    # Our estimated hypervolume is the average of the hyperoctants.
    # (#points, #vectors) --> (#vectors,)
    return jnp.mean(hyper_octant_volume, axis=0)


def _batch_size(points, max_ram, max_batch_size):
    """The largest power-of-2 batch size we can afford."""
    float_size = 4
    num_coordinates = points.shape[0] * points.shape[1]
    estimated_ram_per_batch = float_size * num_coordinates * 2
    max_batch = max_ram / estimated_ram_per_batch
    # Round down to the nearest power of 2.
    max_batch = 2 ** int(math.log2(max_batch))
    max_batch = max(1, max_batch)
    max_batch = min(max_batch, max_batch_size)
    return max_batch


@jax.jit(static_argnames=("num_vectors_pow",))
def prefix_dominated_hypervolume(pareto_points: jnp.ndarray,
                                 origin: jnp.ndarray,
                                 num_vectors_pow: int=15,
                                 random_key=jax.random.key(95310),
                                 max_ram=1e9) -> float:
    """Approximate hypervolume dominated by each prefix of the points, with origin as baseline.

    This is the hypervolume of the locus of points p such that:
    (1) p >= origin in all axes, and
    (2) p <= at least one pareto point in all axes.
    Note that this can be zero if none of pareto_points dominate the origin.

    Args:
        pareto_points: 2D array of (#points, #metrics). For efficiency, these points
        should all be prefix-pareto optimal. (This function won't check.)
        origin: 1D array of (metrics,) used as the baseline against which we
        measure hypervolume.
        num_vectors_pow: We'll generate 2**num_vectors_pow and use each one to
        approximate the hypervolume. We insist on a power of two for convenience
        of batching. More vectors --> paying more compute for more accuracy.
        random_key: The key for generating the vectors.
    """
    if num_vectors_pow < 0:
        raise ValueError(f"num_vectors_pow must be a positive integer, got {num_vectors_pow}.")
    if num_vectors_pow > 20:
        raise ValueError(f"num_vectors_pow={num_vectors_pow} is too high. A reasonable range is " 
                         "10-20 (i.e. 2**10 to 2**20 vectors) to ensure a reasonably accurate " \
                         "approximation for a reasonable amount of compute.")
    origin = jnp.array(origin, dtype=float)
    assert origin.shape == (pareto_points.shape[1],), "origin must have the same shape as a point."
    pareto_points -= origin
    # These points don't contribute to the hypervolume, so zero them out of
    # subsequent calculations.
    bad_points = jnp.any(pareto_points <= 0, axis=-1)
    pareto_points = jnp.where(bad_points.reshape((-1, 1)), 0, pareto_points)
    # Normalize each score (NOT each vector).
    # This should help distribute the vectors over the different dimensions.
    # (By default, the vectors would clump up on the shortest dimensions.)
    pareto_points, multiplier = _normalize_scores(pareto_points)
    # For batching, we insist that the number of vectors is a power of 2.
    num_vectors = 2 ** int(num_vectors_pow)
    num_dimensions = pareto_points.shape[1]
    vectors = jnp.abs(jax.random.normal(shape=(num_vectors, num_dimensions), key=random_key))
    vectors = _normalize_vectors(vectors)

    # To fit RAM, we can't just cross num_vectors against pareto_points.
    # Instead, we'll pick a power-of-two batch_size that fits in RAM.
    batch_size = _batch_size(pareto_points, max_ram, num_vectors)
    num_batches = num_vectors // batch_size

    def average_hypervolume(i, val):
        vector_slice = jax.lax.dynamic_slice_in_dim(
            vectors,
            start_index=i * batch_size,
            slice_size=batch_size)

        new_hypervolume = _prefix_dominated_hypervolume(pareto_points, vector_slice)
        return val + new_hypervolume / num_batches

    prefix_hypervolumes = jax.lax.fori_loop(
        0, num_batches,
        body_fun=average_hypervolume,
        init_val=jnp.zeros(len(pareto_points), dtype=float))
    # When we normalized the metrics, we divided the hypervolume by the normalization factors.
    # Now we multiply by those factors to restore the true hypervolume.
    return prefix_hypervolumes * multiplier
