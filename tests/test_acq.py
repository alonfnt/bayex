from functools import partial

import jax
import jax.numpy as jnp
import pytest

import bayex

KEY = jax.random.PRNGKey(42)
SEED = 42


def f(x):
    return jnp.sin(x) - x / (jnp.cos(x) + 1) + x ** 3 - (x + 1) ** 4


TARGET = -0.3747
bounds = dict(x=(-2, 2))

optim = partial(bayex.optim, f, constrains=bounds, seed=SEED, n=10, n_init=5)


def test_acq_ei():
    param = optim(acq=bayex.ACQ.EI)
    assert jnp.allclose(TARGET, param.target, atol=1e-02)


def test_acq_poi():
    param = optim(acq=bayex.ACQ.POI, n=15)
    assert jnp.allclose(TARGET, param.target, atol=1e-02)


def test_acq_ucb():
    param = optim(acq=bayex.ACQ.UCB, n=15)
    assert jnp.allclose(TARGET, param.target, atol=1e-02)


def test_acq_lcb():
    param = optim(acq=bayex.ACQ.LCB, n=15)
    assert jnp.allclose(TARGET, param.target, atol=1e-02)


def test_acq_ei_custom_xi():
    param = optim(acq=bayex.ACQ.EI, xi=0.02)
    assert jnp.allclose(TARGET, param.target, atol=1e-02)


def test_wrong_acq():
    with pytest.raises(ValueError):
        assert optim(acq="sr", xi=0.02)
