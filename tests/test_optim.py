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
    target_found = f(param)
    assert jnp.allclose(TARGET, target_found, atol=1e-02)


def test_2D_optim():
    """
    Complete test of the optim function for 2 dimensional function.
    """

    def f(x, y):
        return -(y ** 2) - (x - y) ** 2 + 3 * x / y - 2

    TARGET = 2.24936
    bounds = dict(x=(0, 5), y=(1, 4))

    params = bayex.optim(f, constrains=bounds, seed=SEED, n=15, n_init=10)
    target_found = f(*params)
    assert jnp.allclose(TARGET, target_found, rtol=1e-01)
