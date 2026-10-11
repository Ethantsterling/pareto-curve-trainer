# Pareto Optimization for Tiny Interpretable Models.

A JAX library to train small component-based models to optimize multiple objectives over a dataset. These models are interpretable by construction and can enforce guarantees such as monotonicity with respect to particular inputs. Optimization does not use gradients, enabling nondifferentiable models and nondifferentiable training objectives.

Inspired by:
- [PWLFit](https://github.com/google/pwlfit) and associated [curve-GAMs](https://arxiv.org/abs/2101.08393). (I was heavily involved in those projects.)
- [Vizier](https://github.com/google/vizier), especially [NSGA2](https://github.com/google/vizier/blob/main/vizier/_src/algorithms/evolution/nsga2.py).
