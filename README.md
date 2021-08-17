# BAYEX: Bayesian Optimization powered by JAX
Bayex is a plug-and-play python library that provides bayesian global optimization using gaussian processes.
In contrast to existing bayesian optimization libraries, Bayex makes use of JAX as its backend library.

## What is Bayesian Optimization?

Bayesian Optimization (BO) methods are useful for optimizing functions that are expen- sive to evaluate, lack an analytical expression and whose evaluations can be contaminated by noise. These methods rely on a probabilistic model of the objective function, typically a Gaussian process (GP), upon which an acquisition function is built. The acquisition function guides the optimization process and measures the expected utility of performing an evaluation of the objective at a new point. 

## Why JAX?
Using JAX as a backend removes some of the limitations found on Python, as it gives us direct mapping to the XLA compiler.

XLA compiles and runs the JAX code into several architectures such as CPU, GPU and TPU without hassle. But the device agnostic approach is not the reason to back XLA for future scientific programs. XLA provides with optimizations under the hood such as Just-In-Time compilation and automatic parallelization that make Python (with a NumPy-like approach) a suitable candidate on some High Performance Computing scenarios.

Additionally, JAX provides Python code with automatic differentation, which helps indentify the conditions that maximize the acquisition function.


## Installation
Bayex can be installed using PyPI via `pip`:
```
pip install bayex
```

## Getting Started
```python
from bayex import BayesianOptimizer

def f(x, y):
    return -y ** 2 - (x - y) ** 2 + 3 * x / y - 2

constrains = {'x': (-10, 10), 'y': (0, 10)}
optimizer = BayesianOptimizer(f, constrains=constrains)

(x,y), g = optimizer.run(n=10, n_init=4)
```
## Contribute and Building from source

## Planned Featrues
- [ ] Optimization on real numbers domain.
- [ ] Integer parameters support.
- [ ] Categorical Variables 
- [ ] Automatic Parallelization on XLA Devices.

## Citation
To cite this repository
```
@software{
  author = {Albert Alonso},
  title = {{Bayex}: Bayesian Global Optimization with JAX tool for {P}ython+{N}um{P}y programs},
  url = {http://github.com/alonfnt/bayex},
  version = {0.0.0},
  year = {2021},
}
```
## References
