"""Tests for the Pwlcurve component."""
# pylint: disable=missing-function-docstring
# pylint: disable=wrong-import-position
# For most tests, it's quicker and cleaner to lock JAX to CPU.
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax  
import jax.numpy as jnp
import pareto_curve_trainer.component_utils as cu
from pareto_curve_trainer.component_utils import Mono
from pareto_curve_trainer import transforms
from pareto_curve_trainer.transforms import Identity, Log, Symlog1p
from pareto_curve_trainer.component_pwlcurve import (
    jax_curve_slopes, jax_curve_eval, jax_curves_slopes, jax_curves_eval, Pwlcurve,
    pwlcurve_from_data)

# Ensure jittability in these tests.
Pwlcurve = jax.jit(Pwlcurve)

xs = jax.random.uniform(key=jax.random.key(123), shape=5)
curves_xs = jnp.sort(jax.random.uniform(key=jax.random.key(124), shape=(2, 4)))
curves_ys = jax.random.uniform(key=jax.random.key(125), shape=(2,4))

_expected_slopes = jnp.array([jax_curve_slopes(cx, cy) for
                             cx, cy in zip(curves_xs, curves_ys)])
_expected_evals = jnp.array([jax_curve_eval(cx, cy, xs) for
                             cx, cy in zip(curves_xs, curves_ys)])


def test_jax_curves_slopes():
    assert (jax_curves_slopes(curves_xs, curves_ys)[:, :-1] == _expected_slopes).all()


def test_jax_curves_eval():
    _actual_evals = jax_curves_eval(curves_xs, curves_ys, xs)
    assert jnp.allclose(_actual_evals, _expected_evals)


def _is_mono_up(a: jnp.array) -> bool:
    return (a.sort(axis=-1) == a).all()


def _is_mono_down(a: jnp.array) -> bool:
    return (a.sort(axis=-1, descending=True) == a).all()


def _is_nonmono(a: jnp.array) -> bool:
    return not _is_mono_up(a) and not _is_mono_down(a)


def test_mono_sort():
    down_ys0, up_ys1 = cu.vectorized_conditional_mono_sort(
        curves_ys, mono=jnp.array([Mono.DOWN, Mono.UP]))
    nonmono_ys0, nonmono_ys1 = cu.vectorized_conditional_mono_sort(
        curves_ys, mono=jnp.array([Mono.NON_MONO, 0]))
    assert _is_mono_down(down_ys0)
    assert _is_mono_up(up_ys1)
    # Weakness: this could fail if the initial curves happened to be nonmono.
    # With this random key, they happen not to be, but that wasn't guaranteed.
    assert _is_nonmono(nonmono_ys0)
    assert _is_nonmono(nonmono_ys1)


test_pwlcurve = Pwlcurve(
    xs=curves_xs,
    ys=curves_ys,
    xmin=jnp.array([0, -1.]),
    xmax=jnp.array([1, 2.]),
    ymin=jnp.array([0, -1.]),
    ymax=jnp.array([1, 2.]),
    # Note that monotonicity is enforced when perturbing/randomizing,
    # but not on curve creation.
    mono=jnp.array([Mono.DOWN, Mono.NON_MONO]),
)


def test_perturbed():
    perturbed_curve_1 = test_pwlcurve.perturbed(scale=0.1, key=jax.random.key(12345))
    perturbed_curve_2 = test_pwlcurve.perturbed(scale=0.01, key=jax.random.key(12345))
    # Perturbing doesn't change the constraints.
    assert perturbed_curve_1.constraints_equal(test_pwlcurve)
    assert perturbed_curve_2.constraints_equal(test_pwlcurve)

    # Perturbing changes the array.
    assert not perturbed_curve_1 == test_pwlcurve
    assert not perturbed_curve_2 == test_pwlcurve

    # Perturbing by a different scale results in a different array, even with the same key.
    assert not perturbed_curve_1 == perturbed_curve_2


def test_randomized():
    randomized_curve = test_pwlcurve.randomized(key=jax.random.key(12345))
    assert randomized_curve.constraints_equal(test_pwlcurve)
    assert not randomized_curve == test_pwlcurve


def test_merge():
    expected_pwlcurve = Pwlcurve(
        xs=jnp.concatenate([curves_xs, curves_xs]),
        ys=jnp.concatenate([curves_ys, curves_ys]),
        xmin=jnp.array([0, -1, 0, -1.]),
        xmax=jnp.array([1, 2, 1, 2.]),
        ymin=jnp.array([0, -1, 0, -1.]),
        ymax=jnp.array([1, 2, 1, 2.]),
        mono=jnp.array([Mono.DOWN, Mono.NON_MONO, Mono.DOWN, Mono.NON_MONO]),
    )
    assert expected_pwlcurve == test_pwlcurve.merge(test_pwlcurve)


def test_num_rows():
    assert test_pwlcurve.num_rows() == 2


def test_pwlcurve_from_data():
    x = jax.random.uniform(shape=100, key=jax.random.key(324))
    ymin, ymax = -2, 4
    for monotonicity in [Mono.NON_MONO, Mono.UP, Mono.DOWN]:
        pwlcurve = pwlcurve_from_data(x, num_segments=5, ymin=ymin, ymax=ymax, mono=monotonicity)
        assert pwlcurve.xs.shape[-1] == 6 == pwlcurve.ys.shape[-1]
        assert pwlcurve.ymin == ymin
        assert pwlcurve.ymax == ymax
        assert pwlcurve.xmin == jnp.min(x)
        assert pwlcurve.xmax == jnp.max(x)
        assert jnp.all(pwlcurve.xmin <= pwlcurve.xs)
        assert jnp.all(pwlcurve.xs <= pwlcurve.xmax)
        assert jnp.all(pwlcurve.ymin <= pwlcurve.ys)
        assert jnp.all(pwlcurve.ys <= pwlcurve.ymax)
        # xs must be strictly monotonic.
        assert jnp.all(pwlcurve.xs[:, :-1] < pwlcurve.xs[:, 1:])

    # y's monotonicity needs to match `mono=`.
    pwlcurve = pwlcurve_from_data(
        x, num_segments=5, ymin=ymin, ymax=ymax, mono=Mono.UP, x_clip_frac=.2)
    assert jnp.all(pwlcurve.ys[:, :-1] <= pwlcurve.ys[:, 1:])
    pwlcurve = pwlcurve_from_data(
        x, num_segments=5, ymin=ymin, ymax=ymax, mono=Mono.DOWN, x_clip_frac=.1)
    assert jnp.all(pwlcurve.ys[:, :-1] >= pwlcurve.ys[:, 1:])


def test_pwlcurve_from_data_transforms():
    uniform_array = jnp.linspace(1.0, 12, num=1000)
    exp_uniform_array = jnp.exp(uniform_array)

    # pwlcurve_from_data should respect an explicitly-chosen transform.
    for chosen_transform in [None, Identity, Log, Symlog1p]:
        pwlcurve = pwlcurve_from_data(uniform_array, num_segments=4, fx=chosen_transform)
        assert pwlcurve.fx is chosen_transform

    # pwlcurve_from_data should infer the best transform from its options.
    pwlcurve = pwlcurve_from_data(uniform_array, num_segments=4)
    assert pwlcurve.fx is Identity
    pwlcurve = pwlcurve_from_data(exp_uniform_array, num_segments=4)
    assert pwlcurve.fx is Log
    # If Log isn't an option, we'll prefer Symlog1p for the exponentation array.
    pwlcurve = pwlcurve_from_data(exp_uniform_array, num_segments=4, fx=(Identity, Symlog1p))
    assert pwlcurve.fx is Symlog1p
    # If values are negative, Log won't be good, so we'll need Symlog1p.
    pwlcurve = pwlcurve_from_data(-exp_uniform_array, num_segments=4)
    assert pwlcurve.fx is Symlog1p
    # If we exclude Symlog1p, Identity is better for negative values than Log.
    pwlcurve = pwlcurve_from_data(-exp_uniform_array, num_segments=4, fx=(Identity, Log))
    assert pwlcurve.fx is Identity


def test_best_transform():
    uniform_array = jnp.linspace(1.0, 12, num=1000)
    log_uniform_array = jnp.log(uniform_array)
    exp_uniform_array = jnp.exp(uniform_array)

    assert transforms.best_transform(uniform_array) is Identity
    # log array is compressed but applying a log-like transform again would just make it worse.
    assert transforms.best_transform(log_uniform_array) is Identity
    # An exp-distributed array clearly benefits from a log transform.
    assert transforms.best_transform(exp_uniform_array) is Log
