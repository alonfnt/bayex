import jax
import bayex
import jax.numpy as jnp


KEY = jax.random.PRNGKey(42)
SEED = 42

def test_1d_optim():
    """
    Complete test of the optim for a very basic 1d function.
    """

    MAX_VAL = -0.3747
    bounds = dict(x=(-2, 2))

    def f(x):
        return jnp.sin(x) - x / (jnp.cos(x) + 1) + x ** 3 - (x + 1) ** 4

    xmax = bayex.optim(f, constrains=bounds, seed=SEED, n=10, n_init=5)
    ymax = f(xmax)
    assert jnp.allclose(MAX_VAL, ymax, atol=1e-02)
