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

    opt = bayex.optimizer(f, bounds=bounds, nmax=200)

    init_key, key = jax.random.split(KEY, 2)
    state, obs = opt.init(init_key, 5)

    @jax.jit
    def bo_sample(carry):
        key, state, observations = carry
        key, sample_key = jax.random.split(key)
        state = opt.update(state, observations)
        new_obs = opt.sample(sample_key, state, observations, acq=bayex.ACQ.EI)
        observations = bayex.add_observable(observations, new_obs)
        return key, state, observations

    key, state, obs = jax.lax.fori_loop(0, 20, lambda _, c: bo_sample(c), (key, state, obs))
    target = jnp.max(obs.target)
    assert jnp.allclose(TARGET, target, atol=1e-02)
