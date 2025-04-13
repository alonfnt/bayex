import pytest
import jax
import jax.numpy as jnp

from bayex.gp import GPParams
from bayex.acq import (
    expected_improvement,
    probability_improvement,
    upper_confidence_bounds,
    lower_confidence_bounds,
)


@pytest.fixture
def gp_inputs():
    x = jnp.linspace(-3, 3, 10)
    y = jnp.sin(x)
    mask = jnp.ones_like(x)
    xt = jnp.linspace(-4, 4, 5)
    params = GPParams(noise=0.1, amplitude=1.0, lengthscale=1.0)
    return xt, x, y, mask, params


@pytest.mark.parametrize("padding", [0, 3])
def test_acquisition_invariance_to_padding(gp_inputs, padding):
    xt, x, y, mask, params = gp_inputs
    ei_ref, _ = expected_improvement(xt, x, y, mask, params)

    x_pad, y_pad, mask_pad = jax.tree.map(lambda t: jnp.pad(t, (0, padding)), (x, y, mask))
    ei_pad, _ = expected_improvement(xt, x_pad, y_pad, mask_pad, params)

    assert ei_pad.shape == ei_ref.shape
    assert jnp.allclose(ei_ref, ei_pad, atol=1e-5)


@pytest.mark.parametrize("acq_fn", [
    expected_improvement,
    probability_improvement,
    upper_confidence_bounds,
    lower_confidence_bounds,
])
def test_acquisition_output_validity(acq_fn, gp_inputs):
    xt, x, y, mask, params = gp_inputs
    acq_vals, (mu, std) = acq_fn(xt, x, y, mask, params)

    assert acq_vals.shape == xt.shape
    assert mu.shape == xt.shape
    assert std.shape == xt.shape

    assert jnp.all(jnp.isfinite(acq_vals))
    assert jnp.all(jnp.isfinite(mu))
    assert jnp.all(jnp.isfinite(std))
