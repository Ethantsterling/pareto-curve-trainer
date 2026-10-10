population_train(
    xs=test_xs,
    population=test_population,
    metrics_fn=test_metrics_fn,
    model_fn=test_model_fn,
    population_size=5,
    num_steps=1000,
    random_key=jax.random.key(5),
)


def test_xs_and_labels(num_training_points):
  test_xs = {
    'curve_x':  jnp.asarray(np.random.uniform(size=num_training_points)),
    'scalar_x1': jnp.asarray(np.random.uniform(size=num_training_points)),
    'scalar_x2': jnp.asarray(np.random.uniform(size=num_training_points)),
    'cat_x': jnp.asarray(np.random.randint(size=num_training_points, low=0, high=3)),
    'cat_multiplier_x': jnp.asarray(np.random.uniform(size=num_training_points)),
    }
  test_labels = test_model_fn(test_target_components, test_xs)
  return test_xs, test_labels

test_xs_50k, test_labels_50k = test_xs_and_labels(50_000)
test_xs_500k, test_labels_500k = test_xs_and_labels(500_000)
test_xs_5M, test_labels_5M = test_xs_and_labels(5_000_000)

final_population, final_scores = population_train(
    xs=test_xs,
    population=test_population,
    metrics_fn=test_metrics_fn,
    model_fn=test_model_fn,
    population_size=50,
    num_steps=100_000,
    random_key=jax.random.key(8),
)
print(final_scores)

# Very close to test_target_components_2
final_population.filter_population(jnp.array([0])).components

test_target_components_2

final_population.filter_population(jnp.array([1])).components
# Very close to test_target_components

test_target_components