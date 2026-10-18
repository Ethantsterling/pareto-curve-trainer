"""Tests for selection.py."""
# pylint: disable=missing-function-docstring
# pylint: disable=protected-access
# pylint: disable=wrong-import-position
# For most tests, it's quicker and cleaner to lock JAX to CPU.
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax  
import jax.numpy as jnp
from pareto_curve_trainer import selection
from pareto_curve_trainer.selection import (
    pareto_rank, pareto_rank_with_exclusion, pareto_domination,
    _min_masked, crowding_distance, select_by, select)

# Tests need to ensure that select is jittable.
select = jax.jit(select, static_argnames='target_size')


def test_simple_cull_indices():
    culled = selection.simple_cull_indices(jnp.array([5,2,3,4,1.]), 3)
    assert jnp.all(culled == jnp.array([0, 2, 3]))


# Vizier code adapted directly to JAX (replacing np with jnp).
# This is not idiomatic JAX and therefore not as performant, but it's useful as a test to ensure
# the new functions have the intended behavior.
def original_pareto_rank(ys: jnp.ndarray) -> jnp.ndarray:
    if ys.shape[0] == 0:
        return jnp.zeros([0])
    dominated = [jnp.all(ys <= r, axis=-1) & jnp.any(r > ys, axis=-1) for r in ys]
    return jnp.sum(jnp.stack(dominated), axis=0)


# Make sure the rewritten version produces the same output.
small_pop = jax.random.uniform(key=jax.random.key(123), shape=(5,2))
big_population = jax.random.uniform(key=jax.random.key(123), shape=(500,4))


def test_pareto_rank_matches():
    assert jnp.all(pareto_rank(small_pop) == original_pareto_rank(small_pop))
    assert jnp.all(pareto_rank(big_population) == original_pareto_rank(big_population))


def test_pareto_rank_empty():
    ys = jnp.array([]).reshape(0, 2)
    assert pareto_rank(ys).shape == (0,)


def test_pareto_rank_single():
    ys = jnp.array([[1.0, 2.0]])
    assert jnp.all(pareto_rank(ys) == jnp.array([0]))


def test_pareto_rank_simple_dominance():
    # P0 dominates P1
    # P0 dominates P2
    # P1 and P2 don't dominate each other.
    ys = jnp.array([[2.0, 3.0], [1.0, 3.0], [2.0, 2.0]])
    assert jnp.all(pareto_rank(ys) == jnp.array([0, 1, 1]))


def test_pareto_rank_no_dominance():
    ys = jnp.array([[1.0, 5.0], [2.0, 4.0], [3.0, 3.0]])
    assert jnp.all(pareto_rank(ys) == jnp.array([0, 0, 0]))


def test_pareto_rank_duplicate_points_do_not_dominate_each_other():
    ys = jnp.array([[2.0, 3.0], [1.0, 2.0], [2.0, 3.0]])
    assert jnp.all(pareto_rank(ys) == jnp.array([0, 2, 0]))


def mask_dominated(dominated, exclusion_mask):
    return jnp.where(exclusion_mask.reshape((-1,1)), dominated, False)


def test_pareto_domination():
    domination = pareto_domination(jnp.array([[1], [2], [3.]]))
    # [i][j] means [i] dominates [j]
    assert jnp.all(domination == jnp.array([
       [False, False, False],
       [True, False, False],
       [True, True, False],
    ]))
    # Falsify all rows where the mask is False.
    masked = mask_dominated(domination, jnp.array([True, False, True]))
    assert jnp.all(masked == jnp.array([
       [False, False, False],
       [False, False, False],
       [True, True, False],
    ]))


def validate_pareto_rank_with_exclusion(ys, mask):
    """Verify that masking works correctly.

    Specifically, crowding_distance(ys, exclusion_mask=mask)[mask] should be
    equivalent to crowding_distance(ys[mask]). We use the exclusion_mask param
    even though it's more complicated, because it lets us operate on fixed_size
    arrays for JAX jitting.

    Note that we have to disable the random tiebreaker to make this true, because
    the random tiebreaker would change sorting.
    """
    masked_result = pareto_rank_with_exclusion(ys, mask)[mask]
    result = pareto_rank(ys[mask])
    assert (masked_result == result).all(), (masked_result, result)


def test_pareto_rank_with_exclusion():
    """Verify that masking works correctly on a variety of examples."""
    _test_size = 50
    for i in range(5):
        _test_key, _mask_key = jax.random.split(jax.random.key(i))
        for num_metrics in range(1, 4):
            _test_ys = jax.random.randint(key=_test_key, shape=[_test_size, num_metrics], minval=-1, maxval=3).astype(float)
            _test_mask = jax.random.randint(_mask_key, _test_size, minval=0, maxval=2).astype(bool)
            validate_pareto_rank_with_exclusion(_test_ys, _test_mask)


def test_min_masked():
    _mask_test_array = jnp.array(
        [
            [1, 5.],
            [3, 4.],
            [2, 1.],
            [5., 2.],
            [6., 3.],
        ])

    # When the mask is all True, nothing should happen.
    assert (_min_masked(_mask_test_array, jnp.ones(len(_mask_test_array), dtype=bool))
            == _mask_test_array).all()
    # When the mask is all false, we expect all nans.
    assert jnp.isnan(_min_masked(_mask_test_array, jnp.zeros(len(_mask_test_array), dtype=bool))).all()

    # Subtlety: if we mask out the minimum value in a column, the masked row's value
    # will actual increase, because it'll be set to the non-masked min.
    # This is intentional with the way crowding_distance gets calculated.
    assert (_min_masked(_mask_test_array, jnp.array([True, True, False, True, True])) ==
            jnp.array([(1, 5.), (3, 4), (1, 2), (5, 2), (6, 3)])).all()

    assert (_min_masked(_mask_test_array, jnp.array([True, True, True, True, False])) ==
            jnp.array([(1, 5.), (3, 4), (2, 1), (5, 2), (1, 1)])).all()

    assert (_min_masked(_mask_test_array, jnp.array([False, False, True, True, True])) ==
            jnp.array([(2, 1), (2, 1), (2, 1), (5, 2), (6, 3)])).all()


def validate_crowding_distance_with_mask(ys, mask):
    """Verify that masking works correctly.

    Specifically, crowding_distance(ys, exclusion_mask=mask)[mask] should be
    equivalent to crowding_distance(ys[mask]). We use the exclusion_mask param
    even though it's more complicated, because it lets us operate on fixed_size
    arrays for JAX jitting.

    Note that we have to disable the random tiebreaker to make this true, because
    the random tiebreaker would otherwise change sorting.
    """
    masked_result = crowding_distance(ys, exclusion_mask=mask, tiebreaker_random_key=None)[mask]
    result = crowding_distance(ys[mask], tiebreaker_random_key=None)
    assert jnp.isclose(masked_result, result).all(), (masked_result, result)


def test_crowding_distance_on_random_examples():
    _test_size = 50
    for i in range(5):
        _test_key, _mask_key = jax.random.split(jax.random.key(i))
        for num_metrics in range(1, 4):
            _test_ys = jax.random.randint(key=_test_key, shape=[_test_size, num_metrics], minval=-1, maxval=3).astype(float)
            _test_mask = jax.random.randint(_mask_key, _test_size, minval=0, maxval=2).astype(bool)
            validate_crowding_distance_with_mask(_test_ys, _test_mask)


def test_select_by():
    assert (select_by(jnp.array([1,3,3,3,5]), target=3) == jnp.array([0, 1, 1, 1, 2])).all()
    assert (select_by(jnp.array([5,1,3,3,3]), target=3) == jnp.array([2, 0, 1, 1, 1])).all()
    # If there's no border then we're all 0's and 2's.
    print(select_by(jnp.array([5,1,2,4,3]), target=3))
    assert (select_by(jnp.array([5,1,2,4,3]), target=3) == jnp.array([2, 0, 0, 2, 2])).all()
    assert (select_by(jnp.array([1,3,3,3,5]), target=10) == jnp.array([2, 2, 2, 2, 2])).all()
    select_by(jnp.array([3,3,3,3,1]), target=2)

    assert (select_by(
        jnp.array([1,3,3,3,5]),
        target=3,
        tiebreaker=jnp.array([50, 5, 4, 4, 0]),
        ) == jnp.array([0, 2, 1, 1, 2])).all()

    assert (select_by(
        jnp.array([1,3,3,3,5]),
        target=3,
        tiebreaker=jnp.array([50, 5, 4, 3, 0]),
        ) == jnp.array([0, 2, 2, 0, 2])).all()


def arrays_same_sorted(xs, ys):
    xs = jnp.sort(jnp.array(xs))
    ys = jnp.sort(jnp.array(ys))
    return jnp.all(xs == ys)


def test_select():
    # Selection by priority and pareto are predictable.
    # Crowding distance is harder to eyeball, especially when it breaks ties pseudorandomly.
    _test_priority_score = jnp.array([1, 2, 3, 4, 5], dtype=float)
    _test_metric_scores = jnp.array([(2, 2), (3, 3), (4, 4), (5, 5), (1,1)], dtype=float)

    _selection = select(scores=_test_metric_scores, target_size=2, priority_score=_test_priority_score)
    assert arrays_same_sorted(jnp.array([3, 4]), _selection), _selection
    _selection = select(scores=_test_metric_scores, target_size=2)
    assert arrays_same_sorted(jnp.array([2, 3]), _selection), _selection

    # Selection crowding distance tests.
    # When every point is pareto, crowding distance favors spreading out.
    # Note that crowding distance measures each points against its neighbors
    # (in each metric), not only against its selected neighbors.
    # Consequently, selection by crowding distance is not guaranteed to select
    # the most spread frontier available from those points.
    _test_metric_scores = jnp.array([(1, 5), (2, 4), (3, 3), (4, 2), (5,1)], dtype=float)
    _selection = select(scores=_test_metric_scores, target_size=2)
    assert arrays_same_sorted(jnp.array([0, 4]), _selection), _selection

    _test_metric_scores = jnp.array([(1, 5), (3, 3), (4, 2), (5,1)], dtype=float)
    _selection = select(scores=_test_metric_scores, target_size=2)
    assert arrays_same_sorted(jnp.array([0, 3]), _selection), _selection

    _test_metric_scores = jnp.array([(1, 5), (3, 3), (4, 2), (5,1)], dtype=float)
    _selection = select(scores=_test_metric_scores, target_size=3)
    assert arrays_same_sorted(jnp.array([0, 1, 3]), _selection), _selection

    _test_metric_scores = jnp.array([(1, 5), (3, 3), (4, 2), (5,1)], dtype=float)
    _test_priority_score = jnp.array([0, 0, 1, 0.])
    _selection = select(scores=_test_metric_scores, priority_score=_test_priority_score, target_size=3)
    assert arrays_same_sorted(jnp.array([0, 2, 3]), _selection), _selection


def test_selection_when_priority_dominates():
    # Selection when priority dominates.
    for i in range(10):
        for num_metrics in range(1, 4):
            # The metric scores don't matter when the priority score overwhelms them.
            _test_metric_scores = jax.random.uniform(shape=(10, num_metrics), key=jax.random.key(i))
            _test_priority_score = jnp.array([-1, 1, -2, 2, -3, 3, -4, 4, -5, 5], dtype=float)
            _selection = select(scores=_test_metric_scores, priority_score=_test_priority_score, target_size=4)
            assert arrays_same_sorted([3, 5, 7, 9], _selection)


def test_pareto_without_priority():
    # 1D pareto with no intersections and no priority expressed.
    _target_size = 4
    for i in range(10):
        _test_metric_scores = jax.random.uniform(shape=(10, 1), key=jax.random.key(i))
        _expected_indices = jnp.argsort(_test_metric_scores.flatten())[-_target_size:]
        _selection = select(scores=_test_metric_scores, target_size=_target_size)
        assert arrays_same_sorted(_expected_indices, _selection), (_expected_indices, _selection)

    _test_metric_scores = jnp.array([(1, 5), (3, 3), (4, 2), (5,1)], dtype=float)
    _selection = select(scores=_test_metric_scores, target_size=2)
    assert arrays_same_sorted(jnp.array([0, 3]), _selection), _selection

    _test_metric_scores = jnp.array([(1, 5), (3, 3), (4, 2), (5,1)], dtype=float)
    _selection = select(scores=_test_metric_scores, target_size=3)
    assert arrays_same_sorted(jnp.array([0, 1, 3]), _selection), _selection
