import jax.numpy as jnp
import pytest
from bayex.src.gp import GPParams, gaussian_process, exp_quadratic, cov

@pytest.mark.parametrize("x1, x2, mask, expected", [
    (jnp.array([0]), jnp.array([0]), 1, 1),
    (jnp.array([0]), jnp.array([1]), 1, jnp.exp(-1)),
    (jnp.array([1]), jnp.array([1]), 0, 0),
])
def test_exp_quadratic(x1, x2, mask, expected):
    result = exp_quadratic(x1, x2, mask)
    assert jnp.isclose(result, expected), f"Expected {expected}, got {result}"

@pytest.mark.parametrize("x1, x2, mask1, mask2, expected_shape", [
    (jnp.linspace(-5, 5, 10), jnp.linspace(-5, 5, 10), jnp.ones(10), jnp.ones(10), (10, 10)),
    (jnp.linspace(-5, 5, 5), jnp.linspace(-5, 5, 10), jnp.ones(5), jnp.ones(10), (5, 10)),
])
def test_covariance_shape(x1, x2, mask1, mask2, expected_shape):
    cov_matrix = cov(x1, x2, mask1, mask2)
    assert cov_matrix.shape == expected_shape, f"Expected shape {expected_shape}, got {cov_matrix.shape}"

@pytest.mark.parametrize("compute_ml, expected_output_type", [
    (False, tuple),  # Expecting mean and std as output
    (True, jnp.ndarray),  # Expecting marginal likelihood as output
])
def test_gaussian_process_output_type(compute_ml, expected_output_type):
    params = GPParams(noise=0.1, amplitude=1.0, lengthscale=1.0)
    x = jnp.linspace(-5, 5, 10)
    y = jnp.sin(x)
    mask = jnp.ones_like(x, dtype=bool)
    xt = jnp.array([0.0]) if not compute_ml else None

    output = gaussian_process(params, x, y, mask, xt, compute_ml=compute_ml)
    assert isinstance(output, expected_output_type), f"Output type mismatch. Expected: {expected_output_type}, got: {type(output)}"
