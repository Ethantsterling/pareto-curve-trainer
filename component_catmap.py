"""A trainable component mapping a categorical input to floating point output."""

# Curve Utils

import jax
import jax.numpy as jnp

from typing import NamedTuple, Self
import component_utils as cu


class internal_catmap(NamedTuple):
  """A categorical map without the necessary parameters for training."""
  ys: jnp.ndarray

  def eval(self, data: jnp.ndarray):
    # data is required to be indices into the ys.
    # The indices apply in the last index.
    return self.ys[..., data]

  def merge(self, other):
    return internal_catmap(cu._2D_merge(self.ys, other.ys))

  def __eq__(self, other: Self):
    return (self.ys == other.ys).all()


def maybe_sorted_1d(a: jnp.array, mono: cu.Mono) -> jnp.ndarray:
  print('mono is', mono)
  if mono == cu.Mono.UP:
    return a.sort()
  if mono == cu.Mono.DOWN:
    return jnp.flip(a.sort())
  return a


def maybe_sorted(a: jnp.ndarray, mono: jax.typing.ArrayLike) -> jnp.ndarray:
  # a is (N, M) and mono (N).
  # print("sorting", a, "with mono", mono)
  if a.ndim == 1:
    return maybe_sorted_1d(a, jnp.squeeze(mono))
  upsorted = a.sort()
  downsorted = jnp.flip(upsorted, axis=-1)
  mono = jnp.reshape(mono, shape=(-1, 1))
  return jnp.where(mono == cu.Mono.UP, upsorted,
                   jnp.where(mono == cu.Mono.DOWN, downsorted, a))


class catmap(NamedTuple):
  # Categories must be indices into y.
  # That requires preconverting string categories into ints, which we'll need
  # to do somewhere.
  map: internal_catmap                           # 1D or 2D of floats
  ymin: jax.typing.ArrayLike                     # 0D or 1D of floats
  ymax: jax.typing.ArrayLike                     # 0D or 1D of floats
  mono: jax.typing.ArrayLike = cu.Mono.NON_MONO  # Of Mono int enums

  def eval(self, data: jnp.ndarray):
    return self.map.eval(data)

  def perturbed(self, scale: float, key: jax.Array) -> tuple[Self, jax.Array]:
    """Returns itself shifted by a random step in [-scale, +scale]."""
    new_ys, subkey = cu.perturb_array(self.map.ys, self.ymin, self.ymax, scale, key)
    new_ys = maybe_sorted(new_ys, self.mono)
    new_map = self._replace(map=internal_catmap(ys=new_ys))
    return new_map, subkey

  def randomized(self, key: jax.Array) -> tuple[Self, jax.Array]:
    """Returns itself with the scalar randomized."""
    new_ys, subkey = cu.randomize_array(self.map.ys, self.ymin, self.ymax, key)
    new_ys = maybe_sorted(new_ys, self.mono)
    new_map = self._replace(map=internal_catmap(ys=new_ys))
    return new_map, subkey

  def merge(self, other) -> Self:
    """We have to control for the weird dimensionality possibilities. """
    map = self.map.merge(other.map)
    # For a clean merge, we need to broadcast the params to the correct dimensionality.
    sdim = (cu._2D(self.map.ys).shape[0],)
    odim = (cu._2D(other.map.ys).shape[0],)

    symin = jnp.broadcast_to(self.ymin, sdim)
    oymin = jnp.broadcast_to(other.ymin, odim)
    ymin = cu._1D_merge(symin, oymin)

    symax = jnp.broadcast_to(self.ymax, sdim)
    oymax = jnp.broadcast_to(other.ymax, odim)
    ymax = cu._1D_merge(symax, oymax)

    smono = jnp.broadcast_to(self.mono, sdim)
    omono = jnp.broadcast_to(other.mono, odim)
    mono = cu._1D_merge(smono, omono)

    return catmap(map=map, ymin=ymin, ymax=ymax, mono=mono)

  def _constraints_equal(self, other: Self):
    return (jnp.all(self.ymin == other.ymin) and
            jnp.all(self.ymax == other.ymax) and
            jnp.all(self.mono == other.mono))

  def __eq__(self, other: Self) -> bool:
    return (self.map == other.map) and self._constraints_equal(other)

  # NamedTuple uses __len__ so we can't override that.
  def num_rows(self) -> int:
    ys = self.map.ys
    return len(ys) if (jnp.ndim(ys) == 2) else 1
