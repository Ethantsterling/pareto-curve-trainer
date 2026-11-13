"""Tests for transforms.py"""

# pylint: disable=missing-function-docstring
# pylint: disable=wrong-import-position
# For most tests, it's quicker and cleaner to lock JAX to CPU.
import os

os.environ["JAX_PLATFORMS"] = "cpu"
import jax.numpy as jnp

from pareto_curve_trainer import transforms
from pareto_curve_trainer.transforms import Identity, Log, Symlog1p

standard_transform_functions = [Identity, Log, Symlog1p]


def test_best_transform():
    uniform_array = jnp.linspace(1.0, 12, num=1000)
    log_uniform_array = jnp.log(uniform_array)
    exp_uniform_array = jnp.exp(uniform_array)

    assert transforms.best_transform(uniform_array) is Identity
    # log array is compressed but applying a log-like transform again would just make it worse.
    assert transforms.best_transform(log_uniform_array) is Identity
    # An exp-distributed array clearly benefits from a log transform.
    assert (
        transforms.best_transform(
            log_uniform_array, transforms=standard_transform_functions + [jnp.exp]
        )
        is jnp.exp
    )

    assert transforms.best_transform(exp_uniform_array) is Log


def test_best_transform_with_negative_values():
    neg_uniform_array = jnp.linspace(-5.0, 12.0, num=1000)
    neg_exp_uniform_array = jnp.sign(neg_uniform_array) * jnp.exp(jnp.abs(neg_uniform_array))

    # We can't use log if some values are <= 0, but we can use symlog1p.
    assert transforms.best_transform(neg_uniform_array) is Identity
    assert transforms.best_transform(neg_exp_uniform_array) is Symlog1p


def test_identity_transform():
    xs = jnp.linspace(-10.0, 10, num=1000)
    assert jnp.array_equal(xs, Identity(xs))
    assert jnp.array_equal(xs, Identity.inverse(xs))


def test_symlog1p_transform():
    xs = jnp.exp(jnp.linspace(-10.0, 10, num=1000))

    assert jnp.allclose(jnp.log1p(xs), Symlog1p(xs))
    assert jnp.allclose(-jnp.log1p(xs), Symlog1p(-xs))

    # Test symmetry.
    assert jnp.allclose(Symlog1p(xs), -Symlog1p(-xs))

    # Test inversion.
    assert jnp.allclose(xs, Symlog1p.inverse(Symlog1p(xs)))
    assert jnp.allclose(-xs, Symlog1p.inverse(Symlog1p(-xs)))


def test_log_transform():
    xs = jnp.exp(jnp.linspace(-10.0, 10, num=1000))

    assert jnp.allclose(jnp.log(xs), Log(xs))
    # Log should support negative values and map them all to -inf.
    assert jnp.allclose(float("-inf"), Log(-xs))

    # Test inversion. Log is invertible for x >= 0.
    assert jnp.allclose(xs, Log.inverse(Log(xs)))
    assert Log.inverse(Log(0)) == 0
