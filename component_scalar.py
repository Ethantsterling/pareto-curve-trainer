"""A trainable scalar 
Training wrappers around 0D, 1D, and 2D functions that include training
constraints such as bounds for parameters, monotonicity, etc.

This one represents a vector of variations of one scalar.
"""
from typing import NamedTuple, Self

import jax
import jax.numpy as jnp

import component_utils as cu


class scalar(NamedTuple):
  s: jax.typing.ArrayLike
  smin: jax.typing.ArrayLike
  smax: jax.typing.ArrayLike

  def perturbed(self, scale: float, key: jax.Array) -> tuple[Self, jax.Array]:
    """Returns itself shifted by a random step in [-scale, +scale]."""
    new_s, subkey = cu.perturb_array(self.s, self.smin, self.smax, scale, key)
    return self._replace(s=new_s), subkey

  def randomized(self, key: jax.Array) -> tuple[Self, jax.Array]:
    """Returns itself with the scalar randomized."""
    new_s, subkey = cu.randomize_array(self.s, self.smin, self.smax, key)
    return self._replace(s=new_s), subkey

  def value(self) -> jnp.ndarray:
    """Convert to a 1 x n array to support operations against 1D arrays."""
    if jnp.ndim(self.s) == 1:
      return jnp.reshape(self.s, shape=(-1,1))
    return self.s

  def merge(self, other) -> Self:
    s = jnp.concatenate([cu._1D(self.s), cu._1D(other.s)])
    smin = jnp.concatenate([cu._1D(self.smin), cu._1D(other.smin)])
    smax = jnp.concatenate([cu._1D(self.smax), cu._1D(other.smax)])
    return scalar(s=s, smin=smin, smax=smax)

  # NamedTuple uses __len__ so we can't override that.
  def num_rows(self) -> int:
    return len(self.s) if jnp.ndim(self.s) else 1

  def __eq__(self, other: Self):
    s_identical = (jnp.shape(self.s) == jnp.shape(other.s)) & jnp.all(self.s == other.s)
    smin_matches = jnp.all(self.smin == other.smin)
    smax_matches = jnp.all(self.smax == other.smax)
    return s_identical & smin_matches & smax_matches

