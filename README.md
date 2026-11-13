# Pareto Optimization for Tiny Interpretable Models.

A JAX library to train small component-based models to optimize multiple objectives over a dataset. These models are interpretable by construction and can enforce guarantees such as monotonicity with respect to particular inputs. Optimization does not use gradients, enabling nondifferentiable models and nondifferentiable training objectives.

Inspired by:
- [PWLFit](https://github.com/google/pwlfit) and associated [curve-GAMs](https://arxiv.org/abs/2101.08393). (I was part of those projects.)
- [Vizier](https://github.com/google/vizier), especially [NSGA2](https://github.com/google/vizier/blob/main/vizier/_src/algorithms/evolution/nsga2.py).

Example Notebooks:
- [UCI Energy Efficiency](https://colab.research.google.com/drive/1Jr-UI1Y-brFHz-_Q_p85TTeMeUZGxGk7?usp=sharing): trading off two objectives on a tiny dataset. Useful introduction.
- [UCI Forest Cover Type](https://colab.research.google.com/drive/1w7NT2Q5_6N4DDJlJnokJkVVBhEej-iFv?usp=sharing): Multiclass classification of one objective on a larger (~500K) dataset. This is more involved since it's multiclass.
