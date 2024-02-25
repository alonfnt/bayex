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

    assert type(opt_state) == bayex.OptimizerState
    assert type(opt_state.gp_state) == bayex.src.gp.GPState # pyright: ignore

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


def test_evaluate_raise_invalid_acq_fun():
    domain = {'x': bayex.domain.Real(-2, 2)}

    # This shouldn't raise an error
    for acq in ['EI',]:
        bayex.Optimizer(domain=domain, acq=acq)

    # But this should!
    for acq in ['random', 'magic']:
        with pytest.raises(ValueError, match=f"Acquisition function {acq} is not implemented"):
            bayex.Optimizer(domain=domain, acq=acq)


def test_2D_rosenbrock_fun():

    def rosenbrock(x, y, a=1, b=100):
        return (a - x)**2 + b*(y - x**2)**2

    # Example usage:
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)

    min_index = np.argmin(Z)
    min_x = X.flatten()[min_index]
    min_y = Y.flatten()[min_index]
    min_value = Z.flatten()[min_index]

    domain = {'x': bayex.domain.Real(-2, 2), 'y': bayex.domain.Real(-2, 2)}
    opt = bayex.Optimizer(domain=domain, maximize=False)

    params = {'x': [0.0, 0.5, 2.0], 'y': [0.0, 0.5, 2.0]}
    ys = [rosenbrock(x, y) for x,y in zip(params['x'], params['y'])]
    opt_state = opt.init(ys, params)

    key = jax.random.key(SEED)
    sample_fn = jax.jit(opt.sample)
    for step in range(100):
        key = jax.random.fold_in(key, step)
        new_params = sample_fn(key, opt_state)
        y = rosenbrock(**new_params)
        print(f'Params: {new_params}, Value: {y} ({opt_state.best_score})')
        opt_state = opt.fit(opt_state, y, new_params)
        if jnp.allclose(opt_state.best_score, min_value, atol=1e-02):
            break
    target = opt_state.best_score
    assert jnp.allclose(min_value, target, atol=1e-02)
    assert jnp.allclose(min_x, opt_state.best_params['x'], atol=1e-01)
    assert jnp.allclose(min_y, opt_state.best_params['y'], atol=1e-01)
