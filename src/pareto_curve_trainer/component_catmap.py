"""A trainable component mapping a categorical input to floating point output."""
import jax
import jax.numpy as jnp
from typing import NamedTuple, Self
import pareto_curve_trainer.component_utils as cu


class InternalCatmap(NamedTuple):
    """A categorical map without the necessary parameters for training."""
    ys: jnp.ndarray

    def eval(self, data: jnp.ndarray):
        """Evaluate the vector of maps on a vector of data."""
        # data is required to be indices into the ys.
        # The indices apply in the last index.
        return self.ys[..., data]

    def merge(self, other):
        """Concatenate the map with another map."""
        return InternalCatmap(cu._2D_merge(self.ys, other.ys))

    def __eq__(self, other: Self):
        return (self.ys == other.ys).all()


class Catmap(NamedTuple):
    """A vectorized, trainable component representing category-to-float maps."""
    # Categories must be indices into y.
    # That requires preconverting string categories into ints, which we'll need
    # to do somewhere.
    map: InternalCatmap                           # 1D or 2D of floats
    ymin: jax.typing.ArrayLike                     # 0D or 1D of floats
    ymax: jax.typing.ArrayLike                     # 0D or 1D of floats
    mono: jax.typing.ArrayLike = cu.Mono.NON_MONO  # Of Mono int enums

    def eval(self, data: jnp.ndarray):
        """Evaluate each map in the component on each entry in the data."""
        return self.map.eval(data)

    def perturbed(self, scale: float, key: jax.Array) -> Self:
        """Returns itself shifted by a random step in [-scale, +scale]."""
        new_ys = cu.perturb_array(self.map.ys, self.ymin, self.ymax, scale, key)
        new_ys = cu.maybe_sorted(new_ys, self.mono)
        new_map = self._replace(map=InternalCatmap(ys=new_ys))
        return new_map

    def randomized(self, key: jax.Array) -> Self:
        """Returns itself with the scalar randomized."""
        new_ys = cu.randomize_array(self.map.ys, self.ymin, self.ymax, key)
        new_ys = cu.maybe_sorted(new_ys, self.mono)
        new_map = self._replace(map=InternalCatmap(ys=new_ys))
        return new_map

    def merge(self, other: Self) -> Self:
        """We have to control for the weird dimensionality possibilities. """
        merged_map = self.map.merge(other.map)
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

        return Catmap(map=merged_map, ymin=ymin, ymax=ymax, mono=mono)

    def constraints_equal(self, other: Self):
        """Check if this Catmap has the same constraint parameters as another."""
        return (jnp.all(self.ymin == other.ymin) and
                jnp.all(self.ymax == other.ymax) and
                jnp.all(self.mono == other.mono))

    def __eq__(self, other: Self) -> bool:
        return (self.map == other.map) and self.constraints_equal(other)

    # NamedTuple uses __len__ so we can't override that.
    def num_rows(self) -> int:
        """The number of categorical maps represented by this component."""
        ys = self.map.ys
        return len(ys) if (jnp.ndim(ys) == 2) else 1


def catmap_from_data(xs, ymin: int=0, ymax: int=1, mono=cu.Mono.NON_MONO):
    """Initialize a catmap that infers its number of categories from the data.
    
    Complication: the user is responsible for converting categories to the index format, where
    the categories are numbered starting from 0 and incrementing up to (#categories - 1)."""
    xs = jnp.array(xs)
    num_uniques = len(jnp.unique(xs))
    ys = jnp.repeat((ymax - ymin) / 2, num_uniques)

    if mono == cu.Mono.UP:
        ys = jnp.linspace(ymin, ymax, num_uniques)
    elif mono == cu.Mono.DOWN:
        ys = jnp.linspace(ymax, ymin, num_uniques)
    ys = ys.reshape((1, -1))
    ymin = jnp.reshape(ymin, shape=(1,))
    ymax = jnp.reshape(ymax, shape=(1,))
    mono = jnp.reshape(mono, shape=(1,))

    return Catmap(InternalCatmap(ys), ymin=ymin, ymax=ymax, mono=mono)
