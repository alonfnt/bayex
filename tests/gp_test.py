import jax
import jax.numpy as jnp
import pytest
from bayex import gp


@pytest.mark.parametrize(
    "x1, x2, mask, expected",
    [
        (jnp.array([0]), jnp.array([0]), 1, 1),
        (jnp.array([0]), jnp.array([1]), 1, jnp.exp(-1)),
        (jnp.array([1]), jnp.array([1]), 0, 0),
    ],
)
def test_exp_quadratic(x1, x2, mask, expected):
    result = gp.exp_quadratic(x1, x2, mask)
    assert jnp.isclose(result, expected), f"Expected {expected}, got {result}"


@pytest.mark.parametrize(
    "x1, x2, mask1, mask2, expected_shape",
    [
        (jnp.linspace(-5, 5, 10), jnp.linspace(-5, 5, 10), jnp.ones(10), jnp.ones(10), (10, 10)),
        (jnp.linspace(-5, 5, 5), jnp.linspace(-5, 5, 10), jnp.ones(5), jnp.ones(10), (5, 10)),
    ],
)
def test_covariance_shape(x1, x2, mask1, mask2, expected_shape):
    cov_matrix = gp.cov(x1, x2, mask1, mask2)
    assert (
        cov_matrix.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {cov_matrix.shape}"


@pytest.mark.parametrize(
    "compute_ml, expected_output_type",
    [
        (False, tuple),  # Expecting mean and std as output
        (True, jnp.ndarray),  # Expecting marginal likelihood as output
    ],
)
def test_gaussian_process_output_type(compute_ml, expected_output_type):
    params = gp.GPParams(noise=0.1, amplitude=1.0, lengthscale=1.0)
    x = jnp.linspace(-5, 5, 10)
    y = jnp.sin(x)
    mask = jnp.ones_like(x, dtype=bool)
    xt = jnp.array([0.0]) if not compute_ml else None

    output = gp.gaussian_process(params, x, y, mask, xt, compute_ml=compute_ml)
    assert isinstance(
        output, expected_output_type
    ), f"Output type mismatch. Expected: {expected_output_type}, got: {type(output)}"


@pytest.mark.parametrize("padding", [0, 1, 5, 10])
def test_masking_in_gaussian_process(padding: int):

    params = gp.GPParams(noise=0.1, amplitude=1.0, lengthscale=1.0)

    x = jnp.linspace(-5, 5, 10)
    y = jnp.sin(x)
    mask = jnp.ones_like(x, dtype=float)
    reference = gp.marginal_likelihood(params, x, y, mask)

    x_pad, y_pad, mask_pad = jax.tree.map(
        lambda x: jnp.pad(x, (0, padding)), (x, y, mask)
    )
    assert len(x_pad) == len(x) + padding, "X should be padded to 10 + padding length"
    assert mask_pad[len(mask):].sum() == 0, "Mask should be zero for padded values"

    output = gp.marginal_likelihood(params, x_pad, y_pad, mask_pad)

    assert jnp.allclose(reference, output), f"Mismatch for padding={padding}"


@pytest.mark.parametrize("padding", [0, 1, 5, 10])
def test_masking_in_prediction(padding: int):

    params = gp.GPParams(noise=0.1, amplitude=1.0, lengthscale=1.0)

    x = jnp.linspace(-5, 5, 10)
    y = jnp.sin(x)
    mask = jnp.ones_like(x)
    xt = jnp.linspace(-6, 6, 20)

    # Reference prediction without padding
    mean_ref, std_ref = gp.predict(params, x, y, mask, xt)

    # Apply padding
    x_pad, y_pad, mask_pad = jax.tree.map(
        lambda a: jnp.pad(a, (0, padding)), (x, y, mask)
    )

    # Prediction with padded inputs
    mean_pad, std_pad = gp.predict(params, x_pad, y_pad, mask_pad, xt)

    assert mean_pad.shape == mean_ref.shape
    assert std_pad.shape == std_ref.shape
    assert jnp.allclose(mean_pad, mean_ref, atol=1e-5), f"Mean mismatch for padding={padding}"
    assert jnp.allclose(std_pad, std_ref, atol=1e-5), f"Std mismatch for padding={padding}"
