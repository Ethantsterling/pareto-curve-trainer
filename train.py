"""### Train"""
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple
from population import Population
from selection import select


def population_train_one_step(
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

    kept_indices = select(combined_population_metrics, target_size=len(population_metrics))
    kept_population = combined_population.filter_population(kept_indices)
    kept_population_metrics = combined_population_metrics[kept_indices]

    # # Jax doesn't know how many True's are in the kept_mask, so it can't
    # # validate the size of kept_population.
    # # I wrote the code so that it's always target_size, but JAX doesn't know that.
    # # Consequently, the 'mask' is actually indices rather than bools.
    # kept_mask = simple_cull_indices(combined_population_metrics, target_size=len(population_metrics))
    # kept_population = combined_population.filter_population(kept_mask)
    # kept_population_metrics = combined_population_metrics[kept_mask]

    # Note: this approach prevents jitted_population_train from tracking the
    # population history, which we might want later.
    # To fix that, we'd want to pass back the combined population and filter.
    # Or else split the one-step training into split and mutate functions.
    # jax.lax.scan keeps a history. For a full history, we'd want to keep
    # the entire history of the population, the entire history of the metrics,
    # and some rendition of the cull history?
    # Keep a history of which trial_ids survived each cull, yes.
    # A history of components should be modest.
    # 100K steps * 10 components * 4 bytes per float * maybe 15 floats per component
    # is about 60M bytes, which is trivial.
    # So, yes, we can track the entire history if we want to. Cool!
    # For now I'm not going to bother, but it shouldn't be very hard.
    return kept_population, kept_population_metrics, new_key, max_trial_id


@jax.jit(static_argnames=['metrics_fn', 'model_fn'])
def jitted_population_train(
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
    """Returns the trained curves and also the associated metric score."""
    scale = max_scale
    population_predictions = model_fn(population.components, xs)
    population_metrics = jnp.stack(metrics_fn(xs, population_predictions), axis=1)

    max_trial_id = jnp.max(population.trial_ids)

    ## Putting this for-loop in jitted code isn't working, which is interesting.
    def _train_one_step(_, val):
        # Nothing in here actually needs the index, but that probably doesn't matter.
        population, population_metrics, scale, random_key, max_trial_id = val

        new_population, new_population_metrics, new_key, max_trial_id = population_train_one_step(
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
    """This is a non-jitted wrapper that sets everything up."""
    scale_progression = (min_scale / max_scale) ** (population_size/num_steps)

    population, random_key = population.fill_population(fill_size=population_size, key=random_key)

    return jitted_population_train(
        xs=xs,
        population=population,
        metrics_fn=metrics_fn,
        model_fn=model_fn,
        num_steps=num_steps // population_size,
        max_scale=max_scale,
        scale_progression=scale_progression,
        random_key=random_key,
    )
