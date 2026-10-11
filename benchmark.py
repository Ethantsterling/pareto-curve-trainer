"""Benchmark training on CPU and GPU.

GPU should train much faster than CPU.
Benchmarks will (of course) depend on the speed of the system on which they're run."""
import time
import jax
from train import population_train
from example_fixtures import (test_target_components, test_target_components_2, test_population,
                              example_metrics_fn, example_model_fn)


def generate_xs(num_training_points, random_seed=102394):
    """Generate xs suitable for the model in example_fixtures.py"""
    keys = jax.random.split(jax.random.key(random_seed), num=5)
    xs = {
        'curve_x':  jax.random.uniform(keys[0], shape=num_training_points),
        'scalar_x1': jax.random.uniform(keys[1], shape=num_training_points),
        'scalar_x2': jax.random.uniform(keys[2], shape=num_training_points),
        'cat_x': jax.random.randint(keys[3], shape=num_training_points, minval=0, maxval=3),
        'cat_multiplier_x': jax.random.uniform(keys[4], shape=num_training_points),
    }
    xs['label'] = example_model_fn(test_target_components, xs)
    xs['label_2'] = example_model_fn(test_target_components_2, xs)
    return xs


test_xs_100k = generate_xs(100_000)
test_xs_1m = generate_xs(1_000_000)


def is_device_available(device_type):
    """Check if CPU/GPU/TPU is available"""
    if device_type.upper() == 'CPU':
        # In JAX, a detected GPU may mask the CPU, so we have to check this way.
        try:
            jax.devices('cpu')
            return True
        except RuntimeError:
            return False
    return any(device.platform == device_type.lower() for device in jax.devices())


def benchmark_training(xs, num_steps, population_size=50, device_type='CPU'):
    """Measure training time."""
    num_xs = len(next(iter(xs.values())))  # xs is a dictionary of 1D arrays.
    size = f"{num_xs:_} points for {num_steps:_} steps"

    if not is_device_available(device_type):
        print(f"{device_type} not available. Skipping {device_type} benchmark with {size}.\n")
        return

    device = jax.devices(device_type.lower())[0]
    print(f"Training on {device.platform.upper()} with {size}.")

    with jax.default_device(device):
        start_time = time.perf_counter()
        _, learned_scores = population_train(
            xs=xs,
            population=test_population,
            metrics_fn=example_metrics_fn,
            model_fn=example_model_fn,
            population_size=population_size,
            num_steps=num_steps,
            random_key=jax.random.key(5))
        learned_scores.block_until_ready()
        end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Runtime: {elapsed_time:.4f} seconds.\n")


print("""Benchmarking with two MSE objectives and four components:
  * two scalar components,
  * one catmap component,
  * one pwlcurve component.
""")

benchmark_training(
    xs=test_xs_100k,
    num_steps=10_000,
    device_type='CPU',
)

benchmark_training(
    xs=test_xs_100k,
    num_steps=10_000,
    device_type='GPU',
)

benchmark_training(
    xs=test_xs_100k,
    num_steps=100_000,
    device_type='CPU',
)

benchmark_training(
    xs=test_xs_100k,
    num_steps=100_000,
    device_type='GPU',
)

print("Not training on CPU with 1M points; it would be slow.\n")
benchmark_training(
    xs=test_xs_1m,
    num_steps=100_000,
    device_type='GPU',
)
