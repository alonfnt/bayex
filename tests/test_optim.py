import jax
import jax.numpy as jnp

import bayex

KEY = jax.random.PRNGKey(42)
SEED = 42


def test_1D_optim():
    """
    Complete test of the optim for a very basic 1d function.
    """

    def f(x):
        return jnp.sin(x) - x / (jnp.cos(x) + 1) + x ** 3 - (x + 1) ** 4

    TARGET = -0.3747
    bounds = dict(x=(-2, 2))

    param = bayex.optim(f, constrains=bounds, seed=SEED, n=10, n_init=5)
    assert jnp.allclose(TARGET, param.target, atol=1e-02)


def test_2D_optim():
    """
    Complete test of the optim function for 2 dimensional function.
    """

    def f(x, y):
        return -(y ** 2) - (x - y) ** 2 + 3 * x / y - 2

    TARGET = 2.24936
    bounds = dict(x=(0, 5), y=(1, 4))

    params = bayex.optim(f, constrains=bounds, seed=SEED, n=15, n_init=10)
    assert jnp.allclose(TARGET, params.target, rtol=1e-01)


def test_optim_params_correct_output():
    def f(x, y, z):
        return -(y ** 2) - (x - y) ** 2 + 3 * z / y - 2

    bounds = dict(x=(0, 5), y=(1, 4), z=(1, 20))

    param = bayex.optim(f, constrains=bounds, seed=SEED, n=2, n_init=2)
    assert type(param.target) == float
    assert type(param.parameters) == dict
    assert len(param.parameters) == len(bounds)


def test_cast_to_int():
    def f(x, y, z):
        return -(y ** 2) - (x - y) ** 2 + 3 * z / y - 2

    bounds = dict(x=(0, 5), y=(1, 4), z=(1, 20))
    ctypes = dict(z=int)

    param = bayex.optim(
        f, constrains=bounds, seed=SEED, n=2, n_init=2, ctypes=ctypes
    )
    assert int(param.parameters["z"]) == param.parameters["z"]
