"""A class to hold a population of models that represent a slice of trials in a training run.

Instead of tracking a single line of descent, we'll fork many different ways and cull branches.
Heavily inspired by Vizier (https://github.com/google/vizier).
"""
from typing import NamedTuple, Self, Union
import jax
import jax.numpy as jnp


def _extend_array(arr: jax.typing.ArrayLike, length: int) -> jax.Array:
    new_shape = (length,) + jnp.shape(arr)[1:]
    return jnp.resize(arr, new_shape)

# Note: this could get weird if components with only one value are allowed to
# be a lower dimension. For simplicity we shouldn't do that.
def _extend_component(c, length: int):
    return jax.tree_util.tree_map(lambda arr: _extend_array(arr, length), c)


class Population(NamedTuple):
    """A collection of variations of the same model for population training."""
    trial_ids: jax.typing.ArrayLike  # These should be in ascending order to avoid headache.
    components: dict

    def num_trials(self) -> int:
        """The number of trials in a Population."""
        return len(self.trial_ids)

    def max_trial_id(self) -> int:
        """The highest trial ID in the Population."""
        return jnp.max(self.trial_ids)

    def stack(self, other: Self) -> Self:
        """Concatenate two same-shaped populations."""
        # The iterable is compile-time constant so this is fine.
        components = {name: self.components[name].merge(other.components[name]) for name in self.components}
        trial_ids = jnp.concatenate([self.trial_ids, other.trial_ids])
        return Population(trial_ids=trial_ids, components=components)

    def _new_trial_ids(self, first_id: Union[int, None]=None, increment_index: bool=True) -> jax.typing.ArrayLike:
        if first_id is not None:
            return first_id + jnp.arange(self.num_trials())
        if increment_index:
            return self.max_trial_id() + jnp.arange(1, 1 + self.num_trials())
        return self.trial_ids

    def perturbed(self,
                  scale: float,
                  key: jax.Array,
                  first_id: Union[int, None] = None,
                  increment_index:bool = True) -> tuple[Self, jax.Array]:
        """Returns a copy with each component perturbed.

        To determine trial_ids, we can duplicate trial_ids.
        Or we can increment from self.max_trial_id().
        Or we can start from a specified first_id and increment from there.
        """
        new_trial_ids = self._new_trial_ids(first_id=first_id, increment_index=increment_index)
        key, *subkeys = jax.random.split(key, len(self.components) + 1)
        new_components = {name: component.perturbed(scale=scale, key=key_i) for
                          (name, component), key_i in zip(self.components.items(), subkeys)}

        return Population(trial_ids=new_trial_ids, components=new_components), key

    def randomized(self, key: jax.Array, increment_index: bool=True) -> tuple[Self, jax.Array]:
        """Returns a copy with each component randomized."""
        new_trial_ids = self._new_trial_ids(increment_index=increment_index)
        key, *subkeys = jax.random.split(key, len(self.components) + 1)
        new_components = {name: component.randomized(key=key_i) for
                          (name, component), key_i in zip(self.components.items(), subkeys)}
        return Population(trial_ids=new_trial_ids, components=new_components), key

    def fill(self, fill_size: int, key: jax.Array) -> tuple[Self, jax.Array]:
        """Returns a new population padded with randomized entries."""
        # This can be very similar to perturbed, except that we build a random new
        # version and end with a merge.
        fill_amount = fill_size - self.num_trials()
        new_trial_ids = jnp.arange(1, 1 + fill_amount) + self.max_trial_id()

        key, *subkeys = jax.random.split(key, len(self.components) + 1)
        # Extend each component to the desired length and then randomize it.
        new_components = {name: _extend_component(component, fill_amount).randomized(key=key_i)
                          for (name, component), key_i in zip(self.components.items(), subkeys)}
        population_extension = Population(trial_ids=new_trial_ids, components=new_components)
        # Only the padded entries (the population_extension) are randomized.
        # The original entries are preserved in case they were important.
        return self.stack(population_extension), key

    def filter(self, filter_array: jnp.ndarray) -> Self:
        """Return only trials matching the filter, which can be indices or a mask."""
        new_trial_ids = self.trial_ids[filter_array]
        new_components = jax.tree.map(lambda node: node[filter_array], self.components)
        return self._replace(trial_ids=new_trial_ids, components=new_components)

    def __str__(self):
        trial_ids_str = f'Population trial_ids: {self.trial_ids}'
        components_strs = [f'{name}: {component}' for name, component in self.components.items()]
        return '\n'.join([trial_ids_str] + components_strs)

    def __eq__(self, other):
        return ((self.num_trials() == other.num_trials()) &
                jnp.all(self.trial_ids == other.trial_ids) &
                (self.components == other.components))


# This would be cleaner as a classmethod on Population, but that gives JAX trouble.
def from_components(components: dict):
    """Create a Population from a list of components."""
    component_list = list(components.values())
    size = component_list[0].num_rows()
    for c in component_list:
        assert c.num_rows() == size
    trial_ids = jnp.arange(1, size + 1)
    return Population(trial_ids=trial_ids, components=components)
