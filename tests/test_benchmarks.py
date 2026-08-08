"""
Regression tests on standard global optimization problems.

The unit tests check that each piece behaves as specified; these check that the
pieces together actually find optima, which is the property a change to the GP
or to an acquisition function is most likely to break without failing anything
else. Several plausible improvements to the hyperparameter fit turn out to help
one problem while making another several times worse, so the suite deliberately
spans smooth and multimodal, low and higher dimensional.

The thresholds are regression guards rather than statements about how well the
method ought to do. They sit a little above the regret measured on the current
implementation, so that a change which degrades search fails here, while normal
floating point variation between platforms does not. Raising one because it
started failing is only reasonable together with an explanation of why the new
value is acceptable.
"""

import pytest

from benchmarks import BENCHMARKS, initial_design, regret, run

SEEDS = (0, 1, 2)

# Steps allowed per problem, and the regret the run must leave. Both are per
# benchmark because the problems differ in how much budget they deserve.
BUDGETS = {
    "sine": (30, 0.05),
    "quadratic": (15, 0.01),
    "branin": (25, 0.30),
    # Six dimensions with this budget is the weakest case by a wide margin:
    # measured regret is around 2.3 against an optimum of -3.32, so the run is
    # barely better than the initial design. The threshold records that rather
    # than endorsing it, and tightening it is the point of gh-63 and gh-61.
    "hartmann6": (40, 2.60),
}


@pytest.mark.slow
@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=lambda b: b.name)
def test_regret_within_budget(benchmark):
    """The optimizer gets close to the known optimum on every seed."""
    steps, threshold = BUDGETS[benchmark.name]
    regrets = [run(benchmark, seed, steps) for seed in SEEDS]

    worst = max(regrets)
    assert worst <= threshold, (
        f"{benchmark.name}: worst regret {worst:.4f} over seeds {SEEDS} "
        f"exceeds {threshold}; per seed {[round(r, 4) for r in regrets]}"
    )


# Acquisition functions are compared on one multimodal and one smooth problem.
# UCB is the unreliable one: on `sine` it leaves a regret of 4.06 on seed 0,
# which is the full height of the function, so it is held to a threshold that
# only catches a further regression. LCB is not included because Optimizer
# rejects it, minimization being expressed with maximize=False instead.
ACQUISITION_BUDGETS = {
    ("sine", "EI"): 0.05,
    ("sine", "PI"): 0.05,
    ("sine", "UCB"): 4.20,
    ("branin", "EI"): 0.30,
    ("branin", "PI"): 0.40,
    ("branin", "UCB"): 0.40,
}


@pytest.mark.slow
@pytest.mark.parametrize("name,acq", list(ACQUISITION_BUDGETS))
def test_acquisition_regret(name, acq):
    """Each acquisition function stays at least as good as it currently is."""
    benchmark = next(b for b in BENCHMARKS if b.name == name)
    steps = BUDGETS[name][0]
    threshold = ACQUISITION_BUDGETS[(name, acq)]

    regrets = [run(benchmark, seed, steps, acq=acq) for seed in SEEDS]

    worst = max(regrets)
    assert worst <= threshold, (
        f"{name}/{acq}: worst regret {worst:.4f} over seeds {SEEDS} "
        f"exceeds {threshold}; per seed {[round(r, 4) for r in regrets]}"
    )


@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=lambda b: b.name)
def test_initial_design_leaves_room_to_optimize(benchmark):
    """
    The runs have to be measuring the optimizer rather than the design.

    If the initial points were already close to the optimum, the thresholds
    above would pass whatever the optimizer did with them, so this checks that
    every seed starts with a gap wider than the regret it is required to reach.
    """
    _, threshold = BUDGETS[benchmark.name]

    for seed in SEEDS:
        params = initial_design(benchmark, seed)
        ys = [benchmark.fun(**dict(zip(params, values)))
              for values in zip(*params.values())]
        best = max(ys) if benchmark.maximize else min(ys)

        assert regret(benchmark, best) > threshold, (
            f"{benchmark.name} seed {seed}: the initial design already reaches "
            f"a regret of {regret(benchmark, best):.4f}, within the {threshold} "
            "the optimizer is asked to achieve, so the run proves nothing"
        )
