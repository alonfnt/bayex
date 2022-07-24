import jax
import jax.numpy as jnp

import bayex

KEY = jax.random.PRNGKey(42)
SEED = 42


def test_1D_optim():
    def f(x):
        return jnp.sin(x) - x / (jnp.cos(x) + 1) + x**3 - (x + 1) ** 4

    TARGET = -0.3747
    bounds = dict(x=(-2, 2))

    opt = bayex.optimizer(f, bounds=bounds, max_samples=200)

    init_key, key = jax.random.split(KEY, 2)
    state, obs = opt.init(init_key, 5)

    @jax.jit
    def optimize(carry):
        key, state, observations = carry
        key, sample_key = jax.random.split(key)
        state = opt.fit(state, observations)
        new_obs = opt.sample(sample_key, state, observations)
        observations = bayex.add_observable(observations, new_obs)
        return key, state, observations

    key, state, obs = jax.lax.fori_loop(0, 10, lambda _, c: optimize(c), (key, state, obs))

    target = jnp.max(obs.outputs)
    assert jnp.allclose(TARGET, target, atol=1e-02)
