# BAYEX: Bayesian Optimization powered by JAX
[![tests](https://github.com/alonfnt/bayex/actions/workflows/tests.yml/badge.svg)](https://github.com/alonfnt/bayex/actions/workflows/tests.yml)

Bayex is a high performance Bayesian global optimization library using Gaussian processes.
In contrast to existing Bayesian optimization libraries, Bayex is designed to use JAX as its backend.

Instead of relaying on external libraries, Bayex only relies on JAX and its custom implementations, without requiring importing massive libraries such as `sklearn`.

## What is Bayesian Optimization?

Bayesian Optimization (BO) methods are useful for optimizing functions that are expensive to evaluate, lack an analytical expression and whose evaluations can be contaminated by noise.
These methods rely on a probabilistic model of the objective function, typically a Gaussian process (GP), upon which an acquisition function is built.
The acquisition function guides the optimization process and measures the expected utility of performing an evaluation of the objective at a new point.

## Why JAX?
Using JAX as a backend removes some of the limitations found on Python, as it gives us direct mapping to the XLA compiler.

XLA compiles and runs the JAX code into several architectures such as CPU, GPU and TPU without hassle. But the device agnostic approach is not the reason to back XLA for future scientific programs. XLA provides with optimizations under the hood such as Just-In-Time compilation and automatic parallelization that make Python (with a NumPy-like approach) a suitable candidate on some High Performance Computing scenarios.

Additionally, JAX provides Python code with automatic differentiation, which helps identify the conditions that maximize the acquisition function.


## Installation
Bayex can be installed using [PyPI](https://pypi.org/project/bayex/) via `pip`:
```
pip install bayex
```
or from GitHub directly
```
pip install git+git://github.com/alonfnt/bayex.git
```
## Getting Started
```python
import bayex

def f(x, y):
    return -y ** 2 - (x - y) ** 2 + 3 * x / y - 2

constrains = {'x': (-10, 10), 'y': (0, 10)}
optim_params = bayex.optim(f, constrains=constrains, seed=42, n=10)
```
showing the results can be done with
```python
>> bayex.show_results(optim_params, min_len=13)
   #sample      target          x            y
      1        -9.84385      2.87875      3.22516
      2        -307.513     -6.13013      8.86493
      3        -19.2197      6.8417       1.9193
      4        -43.6495     -3.09738      2.52383
      5        -58.9488      2.63803      6.54768
      6        -64.8658      4.5109       7.47569
      7        -78.5649      6.91026      8.70257
      8        -9.49354      5.56705      1.43459
      9        -9.59955      5.60318      1.39322
     10        -15.4077      6.37659      1.5895
     11        -11.7703      5.83045      1.80338
     12        -11.4169      2.53303      3.32719
     13        -8.49429      2.67945      3.0094
     14        -9.17395      2.74325      3.11174
     15        -7.35265      2.86541      2.88627
```
we can then obtain the maximum value found using
```python
>> optim_params.target
-7.352654457092285
```
as well as the input parameters that yield it
```python
>> optim_params.params
{'x': 2.865405, 'y': 2.8862667}
```

## Contributing
Everyone can contribute to Bayex and we welcome pull requests as well as raised issues.
In order to contribute code, one must begin by forking the repository. This creates a copy of the repository on your account.

Bayex uses poetry as a packaging and dependency manager. Hence, once you have cloned your repo on your own machine, you can use
```
poetry install
```
to install on the dependencies needed.
You should start a new branch to write your changes on
```
git checkout -b name-of-change
```
or
```
git branch name-of-change
git checkout name-of-change
```

It is welcome if PR are composed of a single commit, to keep the feature <-> commit balance.
Please, when writing the commit message, try follow the [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) specitifcation.
Once you have made your changes and created your commit, it is recommended to run the pre-commit checks.
```
pre-commit run --all
```
as well as the tests to make sure everything works
```
pytest tests/
```

Remember to amend your current commit with the fixes if any of the checks fails.

## Planned Features
- [x] Optimization on continuos domains.
- [ ] Integer parameters support.
- [ ] Categorical Variables
- [ ] Automatic Parallelization on XLA Devices.

## Citation
To cite this repository
```
@software{
  author = {Albert Alonso}
  title = {{Bayex}: Bayesian Global Optimization with JAX tool for {P}ython programs},
  url = {http://github.com/alonfnt/bayex},
  version = {0.1.0.alpha0},
  year = {2021},
}
```
## References
1. [A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811)
2. [BayesianOptimization Library](https://github.com/fmfn/BayesianOptimization)
3. [JAX: Autograd and XLA](https://github.com/google/jax)
