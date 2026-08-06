import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bayex

KEY = jax.random.key(42)
SEED = 42


def test_1D_optim():

    def f(x):
        return jnp.sin(x) - 0.1 * x**2

    TARGET = np.max(f(np.linspace(-2, 2, 1000)))

    domain = {'x': bayex.domain.Real(-2, 2)}

    opt = bayex.Optimizer(domain=domain, maximize=True)

    # Evaluate 3 times the function
    params = {'x': [0.0, 1.0, 2.0]}
    ys = [f(x) for x in params['x']]
    opt_state = opt.init(ys, params)

    assert opt_state.best_score == np.max(ys)
    assert opt_state.best_params['x'] == params['x'][np.argmax(ys)]

    assert type(opt_state) == bayex.optimizer.OptimizerState
    assert type(opt_state.gp_params) == bayex.gp.GPParams # pyright: ignore

    assert opt_state.params['x'].shape == (10,)
    assert opt_state.ys.shape == (10,)
    assert opt_state.mask.shape == (10,)

    assert np.allclose(opt_state.params['x'][:3], params['x'])
    assert np.allclose(opt_state.ys[:3], ys)
    assert np.allclose(opt_state.mask[:3], [True, True, True])
    assert np.allclose(opt_state.mask[3:], [False] * 7)

    key = jax.random.key(SEED)

    sample_fn = jax.jit(opt.sample)
    for step in range(100):
        key = jax.random.fold_in(key, step)
        new_params = sample_fn(key, opt_state)
        y = f(**new_params)
        opt_state = opt.fit(opt_state, y, new_params)
        if jnp.allclose(opt_state.best_score, TARGET, atol=1e-03):
            break
    target = opt_state.best_score
    assert jnp.allclose(TARGET, target, atol=1e-02)


def test_1D_optim_on_wide_domain():
    # The Gaussian Process sees the transformed inputs, so a wide domain has to
    # work just as well as a narrow one. This is the function of test_1D_optim
    # stretched by a factor of 100.
    def f(x):
        return jnp.sin(x / 100) - 0.1 * (x / 100) ** 2

    TARGET = np.max(f(np.linspace(-200, 200, 1000)))

    domain = {'x': bayex.domain.Real(-200, 200)}

    opt = bayex.Optimizer(domain=domain, maximize=True)

    params = {'x': [0.0, 100.0, 200.0]}
    ys = [f(x) for x in params['x']]
    opt_state = opt.init(ys, params)

    key = jax.random.key(SEED)

    # The budget is kept short on purpose: without the transform the kernel
    # underflows on a domain this wide and the search degrades to sampling.
    sample_fn = jax.jit(opt.sample)
    for step in range(20):
        key = jax.random.fold_in(key, step)
        new_params = sample_fn(key, opt_state)
        y = f(**new_params)
        opt_state = opt.fit(opt_state, y, new_params)
    assert jnp.allclose(TARGET, opt_state.best_score, atol=1e-03)


def test_sample_discards_non_finite_candidates(monkeypatch):
    domain = {'x': bayex.domain.Real(-2.0, 2.0)}
    opt = bayex.Optimizer(domain=domain, maximize=True)

    params = {'x': [0.0, 1.0, 2.0]}
    ys = [float(jnp.sin(x)) for x in params['x']]
    opt_state = opt.init(ys, params)

    # Where the acquisition is flat the L-BFGS refinement can diverge, and a
    # single non-finite candidate is enough to win the argmax and from there
    # poison the whole optimizer state.
    monkeypatch.setattr(
        bayex.optimizer, "_optimize_suggestion",
        lambda params, fun, max_iter=10: jnp.full_like(params, jnp.nan),
    )

    new_params = opt.sample(jax.random.key(SEED), opt_state)
    assert jnp.all(jnp.isfinite(new_params['x']))


def test_evaluate_raise_invalid_acq_fun():
    domain = {'x': bayex.domain.Real(-2, 2)}

    # This shouldn't raise an error
    for acq in ['EI', 'PI', 'UCB']:
        bayex.Optimizer(domain=domain, acq=acq)

    # But this should!
    for acq in ['random', 'magic']:
        with pytest.raises(ValueError, match=f"Acquisition function {acq} is not implemented"):
            bayex.Optimizer(domain=domain, acq=acq)


def test_evaluate_raise_on_lcb():
    domain = {'x': bayex.domain.Real(-2, 2)}

    # LCB is meant to be minimized, so the optimizer cannot make use of it.
    with pytest.raises(ValueError, match="LCB cannot be maximized"):
        bayex.Optimizer(domain=domain, acq='LCB')
