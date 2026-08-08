"""
Standard test functions for measuring optimization quality.

Each benchmark carries its own domain and its known optimum, so that a run can
be scored by the regret it leaves rather than by whether it happens to improve.
The functions are the usual ones from the global optimization literature, kept
here so that the numbers can be compared against published results.

All of them are written as minimization problems, which is the convention used
to report their optima, and `run` flips the sign when a benchmark asks to be
maximized.
"""

from typing import Callable, NamedTuple

import numpy as np

import bayex


class Benchmark(NamedTuple):
    """
    A test function together with everything needed to score a run on it.

    Attributes:
        name: Identifier used in test ids and reports.
        fun: Objective, taking one keyword argument per domain entry.
        domain: Mapping of parameter names to their domains.
        optimum: Known optimal value of `fun` over `domain`.
        maximize: Whether `optimum` is a maximum rather than a minimum.
    """
    name: str
    fun: Callable
    domain: dict
    optimum: float
    maximize: bool


def _sine(x):
    # The function used by the README example. Multimodal on [0, 2], with a
    # local maximum at x = 1.49 that is close enough to the global one at
    # x = 1.83 to catch an optimizer that stops exploring too early.
    return -(1.4 - 3 * x) * np.sin(18 * x)


def _quadratic(x, y):
    # Smooth and unimodal. Nothing should ever fail this, so it is the control
    # that catches a change which buys multimodal performance by giving up on
    # the easy case.
    return (x - 0.3) ** 2 + (y + 0.7) ** 2


def _branin(x, y):
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi
    r, s, t = 6.0, 10.0, 1 / (8 * np.pi)
    return a * (y - b * x**2 + c * x - r) ** 2 + s * (1 - t) * np.cos(x) + s


_H_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H_A = np.array([
    [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
    [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
    [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
    [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
])
_H_P = 1e-4 * np.array([
    [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
    [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
    [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
    [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
])


def _hartmann6(**kwargs):
    # The optimizer hands back arrays of shape (1,) while the initial design
    # passes plain floats, so each coordinate is reduced to a scalar first.
    x = np.array([float(np.ravel(kwargs[f"x{i}"])[0]) for i in range(6)])
    inner = np.sum(_H_A * (x - _H_P) ** 2, axis=1)
    return -float(np.dot(_H_ALPHA, np.exp(-inner)))


BENCHMARKS = [
    Benchmark(
        name="sine",
        fun=_sine,
        domain={"x": bayex.domain.Real(0.0, 2.0)},
        optimum=4.101171,
        maximize=True,
    ),
    Benchmark(
        name="quadratic",
        fun=_quadratic,
        domain={"x": bayex.domain.Real(-2.0, 2.0),
                "y": bayex.domain.Real(-2.0, 2.0)},
        optimum=0.0,
        maximize=False,
    ),
    Benchmark(
        name="branin",
        fun=_branin,
        domain={"x": bayex.domain.Real(-5.0, 10.0),
                "y": bayex.domain.Real(0.0, 15.0)},
        optimum=0.397887,
        maximize=False,
    ),
    Benchmark(
        name="hartmann6",
        fun=_hartmann6,
        domain={f"x{i}": bayex.domain.Real(0.0, 1.0) for i in range(6)},
        optimum=-3.322368,
        maximize=False,
    ),
]


def initial_design(benchmark: Benchmark, seed: int, n: int = 5) -> dict:
    """
    Draws the observations used to initialise the GP.

    A run is only meaningful if the optimizer, and not the initial design, is
    what found the optimum, so the design is drawn uniformly and kept small.

    Args:
        benchmark: Benchmark whose domain is sampled.
        seed: Seed for the design, kept separate from the optimizer's key.
        n: Number of initial points.

    Returns:
        Dict mapping each parameter name to a list of `n` values.
    """
    rng = np.random.default_rng(seed)
    return {
        name: list(rng.uniform(dom.lower, dom.upper, size=n))
        for name, dom in benchmark.domain.items()
    }


def regret(benchmark: Benchmark, best_score: float) -> float:
    """
    Distance between the best value found and the known optimum.

    Reported as a non-negative number regardless of the direction of the
    benchmark, so that thresholds read the same way everywhere.
    """
    return abs(float(best_score) - benchmark.optimum)


def run(benchmark: Benchmark, seed: int, steps: int,
        acq: str = "EI", size: int = 5_000) -> float:
    """
    Runs one optimization and returns the regret it leaves.

    Args:
        benchmark: Benchmark to optimize.
        seed: Seed for both the initial design and the optimizer's keys.
        steps: Number of points the optimizer is allowed to sample.
        acq: Acquisition function to use.
        size: Candidates drawn per step. Smaller than the library default to
            keep a full sweep affordable in CI.

    Returns:
        Regret of the best observation made.
    """
    import jax

    optimizer = bayex.Optimizer(
        domain=benchmark.domain, maximize=benchmark.maximize, acq=acq
    )

    params = initial_design(benchmark, seed)
    ys = [benchmark.fun(**dict(zip(params, values)))
          for values in zip(*params.values())]
    opt_state = optimizer.init(ys, params)

    key = jax.random.key(seed)
    for step in range(steps):
        subkey = jax.random.fold_in(key, step)
        new_params = optimizer.sample(subkey, opt_state, size=size)
        y_new = benchmark.fun(**new_params)
        opt_state = optimizer.fit(opt_state, y_new, new_params)

    return regret(benchmark, opt_state.best_score)
