"""Benchmark training on CPU and GPU.

GPU should train much faster than CPU.
Benchmarks will (of course) depend on the speed of the system on which they're run."""
import time
import jax
from pareto_curve_trainer.train import population_train
from pareto_curve_trainer.example_fixtures import (
    test_target_components, test_target_components_2, test_population,
    example_metrics_fn, example_model_fn, generate_xs, make_example_ranked_metrics_fn)


test_xs_10k = generate_xs(10_000)
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


def benchmark_training(xs,
                       num_steps,
                       population_size=50,
                       device_type='CPU',
                       metrics_fn=example_metrics_fn):
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
            metrics_fn=metrics_fn,
            model_fn=example_model_fn,
            population_size=population_size,
            num_steps=num_steps,
            random_key=jax.random.key(5))
        learned_scores.block_until_ready()
        end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Runtime: {elapsed_time:.4f} seconds.\n")


print("""Benchmarking with two group ranking objectives and four components:
  * two scalar components,
  * one catmap component,
  * one pwlcurve component.
""")

# For efficiency, ranking metrics get bound to their data.
ranking_metrics_fn_10k = make_example_ranked_metrics_fn(test_xs_10k)
ranking_metrics_fn_100k = make_example_ranked_metrics_fn(test_xs_100k)
ranking_metrics_fn_1m = make_example_ranked_metrics_fn(test_xs_1m)

benchmark_training(
    xs=test_xs_10k,
    num_steps=10_000,
    device_type='CPU',
    metrics_fn=ranking_metrics_fn_10k,
)

benchmark_training(
    xs=test_xs_100k,
    num_steps=10_000,
    device_type='GPU',
    metrics_fn=ranking_metrics_fn_100k,
)

benchmark_training(
    xs=test_xs_100k,
    num_steps=100_000,
    device_type='GPU',
    metrics_fn=ranking_metrics_fn_100k,
)

benchmark_training(
    xs=test_xs_1m,
    num_steps=100_000,
    device_type='GPU',
    metrics_fn=ranking_metrics_fn_1m,
)

print("""\nBenchmarking with two MSE objectives and four components:
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

print("\nPopulation size 10_000 to make sure we can do that without exceeding memory:")
benchmark_training(
    xs=test_xs_1m,
    num_steps=100_000,
    population_size=10_000,
    device_type='GPU',
)