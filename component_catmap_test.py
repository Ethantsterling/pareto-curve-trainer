# pylint: disable=missing-function-docstring
# pylint: disable=wrong-import-position
# For most tests, it's quicker and cleaner to lock JAX to CPU.
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

from component_utils import Mono
from component_catmap import Catmap, InternalCatmap

# Ensure jittability in these tests.
InternalCatmap = jax.jit(InternalCatmap)
Catmap = jax.jit(Catmap)


_mapdata = jnp.array([3, 2, 1, 0])
internal_catmap_1d = InternalCatmap(jnp.array([1., 2, 3, 2.]))
internal_catmap_2d = InternalCatmap(jnp.array([[1., 2, 3, 2.], [5., 0., 3., 4.]]))

def test_internal_catmap_eval():
    assert (internal_catmap_1d.eval(_mapdata)
            == jnp.array([2., 3., 2., 1.])).all()
    assert (internal_catmap_2d.eval(_mapdata)
            == jnp.array([[2., 3., 2., 1.], [4., 3., 0., 5.]])).all()

def test_internal_catmap_merge():
    assert (internal_catmap_1d.merge(internal_catmap_2d).ys == jnp.array([
        [1., 2, 3, 2.], [1., 2, 3, 2.], [5., 0., 3., 4.]])).all()


_test_map_1d = Catmap(internal_catmap_1d, ymin=0, ymax=5)
_test_map_2d = Catmap(internal_catmap_2d, ymin=jnp.array([0., 0]), ymax=jnp.array([5., 5]))


def test_num_rows():
    assert _test_map_1d.num_rows() == 1
    assert _test_map_2d.num_rows() == 2


def test_catmap_merge():
    expected_merge = Catmap(
        map=internal_catmap_1d.merge(internal_catmap_2d),
        ymin=jnp.array([0, 0, 0]),
        ymax=jnp.array([5, 5, 5]),
        mono=jnp.array([Mono.NON_MONO, Mono.NON_MONO, Mono.NON_MONO]))

    assert _test_map_1d.merge(_test_map_2d) == expected_merge


def test_catmap_eval():
    assert (_test_map_1d.eval(_mapdata) == jnp.array([2., 3., 2., 1.])).all()
    assert (_test_map_2d.eval(_mapdata) == jnp.array([[2., 3., 2., 1.], [4., 3., 0., 5.]])).all()


def test_catmap_perturbed():
    # Perturb with scale=0.0 should change nothing.
    assert _test_map_1d == _test_map_1d.perturbed(scale=0.0, key=jax.random.key(11))[0]
    assert _test_map_2d == _test_map_2d.perturbed(scale=0.0, key=jax.random.key(12))[0]

    # Perturbing should break equality but maintain constraint equality.
    assert _test_map_1d != _test_map_1d.perturbed(scale=0.1, key=jax.random.key(13))[0]
    assert _test_map_2d != _test_map_2d.perturbed(scale=0.1, key=jax.random.key(14))[0]
    assert _test_map_1d.constraints_equal(_test_map_1d.perturbed(scale=0.1, key=jax.random.key(12))[0])
    assert _test_map_2d.constraints_equal(_test_map_2d.perturbed(scale=0.1, key=jax.random.key(12))[0])


# Perturbing should obey monotonicity as assigned.
def _map_is_mono_up(map: jnp.array) -> bool:
  return (map.map.ys.sort(axis=-1) == map.map.ys).all()
def _map_is_mono_down(map: jnp.array) -> bool:
  return (map.map.ys.sort(axis=-1) == jnp.flip(map.map.ys, axis=-1)).all()
def _map_is_nonmono(map: jnp.array) -> bool:
  return not _map_is_mono_up(map) and not _map_is_mono_down(map)


# Validate the monotonicity measures.
assert _map_is_nonmono(_test_map_1d)
assert _map_is_nonmono(_test_map_2d)

assert _map_is_mono_up(Catmap(InternalCatmap(jnp.array([1., 2., 3., 4.])), ymin=0, ymax=5))
assert not _map_is_mono_down(Catmap(InternalCatmap(jnp.array([1., 2., 3., 4.])), ymin=0, ymax=5))
assert not _map_is_mono_up(Catmap(InternalCatmap(jnp.array([4., 3., 2., 1.])), ymin=0, ymax=5))
assert _map_is_mono_down(Catmap(InternalCatmap(jnp.array([4., 3., 2., 1.])), ymin=0, ymax=5))

assert _map_is_nonmono(_test_map_2d)
assert not _map_is_mono_up(_test_map_2d)
assert not _map_is_mono_down(_test_map_2d)

_mono_up_catmap2d = InternalCatmap(jnp.array([[1., 2, 3, 4.], [0., 3., 4., 5.]]))
_mono_up_test_map2d = Catmap(_mono_up_catmap2d, ymin=0, ymax=5)
assert _map_is_mono_up(_mono_up_test_map2d)
_mono_down_catmap2d = InternalCatmap(jnp.array([[4., 3, 2, 1.], [5., 4., 3., 0.]]))
_mono_down_test_map2d = Catmap(_mono_down_catmap2d, ymin=0, ymax=5)
assert _map_is_mono_down(_mono_down_test_map2d)


def test_nonmono_1d():
    nonmono_test_map_1d = _test_map_1d.perturbed(scale=0.1, key=jax.random.key(12))[0]
    assert _map_is_nonmono(nonmono_test_map_1d)

    nonmono_test_map_1d = _test_map_1d._replace(mono=Mono.NON_MONO)
    nonmono_test_map_1d = nonmono_test_map_1d.perturbed(scale=0.1, key=jax.random.key(12))[0]
    assert _map_is_nonmono(nonmono_test_map_1d)

def test_monoup_1d():
    monoup_test_map_1d = Catmap(internal_catmap_1d, ymin=0, ymax=5, mono=Mono.UP)
    monoup_test_map_1d = monoup_test_map_1d.perturbed(scale=0.1, key=jax.random.key(12))[0]
    assert _map_is_mono_up(monoup_test_map_1d)

    monodown_test_map_1d = Catmap(internal_catmap_1d, ymin=0, ymax=5, mono=Mono.DOWN)
    monodown_test_map_1d = monodown_test_map_1d.perturbed(scale=0.1, key=jax.random.key(12))[0]
    assert _map_is_mono_down(monodown_test_map_1d)

def test_nonmono_2d():
    nonmono_test_map_2d = _test_map_2d.perturbed(scale=0.1, key=jax.random.key(12))[0]
    assert _map_is_nonmono(nonmono_test_map_2d)

    nonmono_test_map_2d = _test_map_2d._replace(mono=Mono.NON_MONO)
    nonmono_test_map_2d = nonmono_test_map_2d.perturbed(scale=0.1, key=jax.random.key(12))[0]
    assert _map_is_nonmono(nonmono_test_map_2d)

def test_monoup_2d():
    monoup_test_map_2d = _test_map_2d._replace(mono=Mono.UP)
    monoup_test_map_2d = monoup_test_map_2d.perturbed(scale=0.1, key=jax.random.key(12))[0]
    assert _map_is_mono_up(monoup_test_map_2d)

def test_monodown_2d():
    monodown_test_map_2d = _test_map_2d._replace(mono=Mono.DOWN)
    monodown_test_map_2d = monodown_test_map_2d.perturbed(scale=0.1, key=jax.random.key(12))[0]
    assert _map_is_mono_down(monodown_test_map_2d)

def test_mixed_2d():
    """The case where constraints differ for different rows."""
    internal_map = InternalCatmap(jnp.array([[1., 3, 2.], [5., 0., 4.], [-4, -2, -5]]))
    mixed_map = Catmap(internal_map,
                       ymin=jnp.array([1, 0, -5]),
                       ymax=jnp.array([3, 5, -2]),
                       mono=jnp.array([Mono.UP, Mono.DOWN, Mono.NON_MONO]),
                       )

    mixed_map = mixed_map.perturbed(scale=0.1, key=jax.random.key(12))[0]
    ys = mixed_map.map.ys

    # These should obey the specified monotonicity.
    assert jnp.all(ys[0] == jnp.sort(ys[0]))            # Mono.UP
    assert jnp.all(ys[1] == jnp.flip(jnp.sort(ys[1])))  # Mono.DOWN
    # np.testing.assert_array_equal(ys[0], jnp.sort(ys[0]))            # Mono.UP
    # np.testing.assert_array_equal(ys[1], jnp.flip(jnp.sort(ys[1])))  # Mono.DOWN

    # This should preserve the order of the original ys [-4, -2, -5],
    # since the scale wasn't large enough to change the order.
    assert ys[2][0] < ys[2][1]
    assert ys[2][1] > ys[2][2]
    assert ys[2][0] > ys[2][2]
    assert ys[2][1] > ys[2][0] > ys[2][2]
    