"""A trainable component mapping a categorical input to floating point output."""
import jax
import jax.numpy as jnp
from typing import NamedTuple, Self
import component_utils as cu


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

    def perturbed(self, scale: float, key: jax.Array) -> tuple[Self, jax.Array]:
        """Returns itself shifted by a random step in [-scale, +scale]."""
        new_ys, subkey = cu.perturb_array(self.map.ys, self.ymin, self.ymax, scale, key)
        new_ys = cu.maybe_sorted(new_ys, self.mono)
        new_map = self._replace(map=InternalCatmap(ys=new_ys))
        return new_map, subkey

    def randomized(self, key: jax.Array) -> tuple[Self, jax.Array]:
        """Returns itself with the scalar randomized."""
        new_ys, subkey = cu.randomize_array(self.map.ys, self.ymin, self.ymax, key)
        new_ys = cu.maybe_sorted(new_ys, self.mono)
        new_map = self._replace(map=InternalCatmap(ys=new_ys))
        return new_map, subkey

    def merge(self, other) -> Self:
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
