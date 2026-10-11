"""Multiobjective population training."""
from typing import Dict, Tuple
import jax
import jax.numpy as jnp
from population import Population
from selection import select


def _population_train_one_step(
    *,
    xs: Dict[str, jnp.ndarray],
    population: Population,
    population_metrics: jnp.ndarray,  #  2D (population_size, num_metrics) array
    metrics_fn,  # Function that evalutes and returns our metrics.
    model_fn,
    update_scale: float,
    next_trial_id: int,
    random_key: jax.Array,
    ) -> Tuple[Population, jnp.ndarray, jnp.ndarray, int]:
    """Grows, mutates, and culls the population."""
    new_population, new_key = population.perturbed(update_scale, random_key, first_id=next_trial_id)
    max_trial_id = new_population.max_trial_id()
    new_predictions = model_fn(new_population.components, xs)
    # Combine metrics into (population_size, num_metrics) score array.
    new_population_metrics = jnp.stack(metrics_fn(xs, new_predictions), axis=1)

    combined_population = population.stack(new_population)
    combined_population_metrics = jnp.concatenate([population_metrics, new_population_metrics])

    # We filter by indices instead of boolean mask for ease of jitting.
    # JAX needs to know the size of each array at jitting time. The size of arr[indices] depends
    # only on the size of indices, not the values of the indices contents, which works for jitting.
    # The size of arr[bool_mask] depends on the values in bool_mask, which would break jitting.
    kept_indices = select(combined_population_metrics, target_size=len(population_metrics))
    kept_population = combined_population.filter(kept_indices)
    kept_population_metrics = combined_population_metrics[kept_indices]

    # TODO: use a faster culling selection than `selection.select` when there's just one metric?
    # TODO: to support recording the training history, make this function also return
    # new_population and new_population_metrics?
    # Or split this function into split and mutate functions, and have the higher-level function
    # handle it.
    return kept_population, kept_population_metrics, new_key, max_trial_id


@jax.jit(static_argnames=['metrics_fn', 'model_fn'])
def _jitted_population_train(
    *,
    xs: Dict[str, jnp.ndarray],
    population: Population,  # This must be already filled.
    metrics_fn, #: Metric function that knows the labels.
    model_fn,
    num_steps: int = 100,
    max_scale: float = 0.1,
    scale_progression: float = 1.0,
    # min_scale: float = 0.01,
    random_key: jax.Array = jax.random.key(12345),
    ) -> Tuple[Population, jnp.ndarray]:  # No need to return the updated random key IMO.
    """Returns the trained curves and also the associated metric score. Jitted for performance."""
    scale = max_scale
    population_predictions = model_fn(population.components, xs)
    population_metrics = jnp.stack(metrics_fn(xs, population_predictions), axis=1)

    max_trial_id = jnp.max(population.trial_ids)

    ## Putting this for-loop in jitted code isn't working, which is interesting.
    def _train_one_step(_, val):
        population, population_metrics, scale, random_key, max_trial_id = val

        new_population, new_population_metrics, new_key, max_trial_id = _population_train_one_step(
            xs=xs,
            population=population,
            population_metrics=population_metrics,
            metrics_fn=metrics_fn,
            model_fn=model_fn,
            update_scale=scale,
            next_trial_id=max_trial_id+1,
            random_key=random_key)

        scale *= scale_progression
        return new_population, new_population_metrics, scale, new_key, max_trial_id

    # TODO: use a jax.lax.scan instead to remember and return the entire population history?
    # Maybe lock that behind a flag.
    # If we return the population and metrics in trial number order, and also include an array
    # marking what step each population entry survived until, that would enable more advanced
    # analysis of the training results and the pareto frontier.
    # A population history should take up a modest amount of memory, since the models are small
    # and the history size doesn't depend on the amount of data.
    # 100K trials * ~10 floats per components * ~10 components per model * 4 bytes per float
    # = 40M bytes, which is modest. We'd also need to save metric scores and trial ids, but those
    # are also modest. Even a large training run (1M trials, many components) would take a few GB
    # at most for the history.
    population, population_metrics, scale, random_key, max_trial_id = jax.lax.fori_loop(
        0, num_steps - 1,  # The first batch is the initial population.
        _train_one_step,
        (population, population_metrics, scale, random_key, max_trial_id))

    return population, population_metrics


def population_train(
    *,
    xs: Dict[str, jnp.ndarray],
    population: Population,
    metrics_fn, #: Metric,
    model_fn,
    population_size: int = 50,
    num_steps: int = 10_000,
    max_scale: float = 0.1,
    min_scale: float = 0.01,
    random_key: jax.Array = jax.random.key(12345),
    ) -> Tuple[Population, jnp.ndarray]:
    """A non-jitted wrapper that sets up and runs a population-based training.
    
    This wrapper is probably unnecessary. If it turns out that we don't need any non-jitted
    training preparation, we can push this logic down into jitted_population_train."""
    scale_progression = (min_scale / max_scale) ** (population_size/num_steps)
    population, random_key = population.fill(fill_size=population_size, key=random_key)

    return _jitted_population_train(
        xs=xs,
        population=population,
        metrics_fn=metrics_fn,
        model_fn=model_fn,
        num_steps=num_steps // population_size,
        max_scale=max_scale,
        scale_progression=scale_progression,
        random_key=random_key,
    )
