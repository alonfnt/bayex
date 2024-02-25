import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from bayex.src.gp import GPParams, predict


def expected_improvement(
    x_pred: jnp.ndarray,
    xs: jax.Array,
    ys: jax.Array,
    mask: jax.Array,
    gp_params: GPParams,
    xi: float = 0.01,
    ):
    ymax = jnp.max(ys, where=mask, initial=-jnp.inf)
    mu, std = predict(gp_params, xs, ys, mask, xt=x_pred)
    a = mu - ymax - xi
    z = a / (std + 1e-3)
    ei = a * norm.cdf(z) + std * norm.pdf(z)
    return ei, (mu, std)


def probability_improvement(
    x_pred: jnp.ndarray,
    xs: jax.Array,
    ys: jax.Array,
    mask: jax.Array,
    gp_params: GPParams,
    xi: float = 0.01,
    ):
    y_max = ys.max()
    mu, std = predict(gp_params, xs, ys, mask, xt=x_pred)
    z = (mu - y_max - xi) / std
    return norm.cdf(z), (mu, std)


def upper_confidence_bounds(
    x_pred: jnp.ndarray,
    xs: jax.Array,
    ys: jax.Array,
    mask: jax.Array,
    gp_params: GPParams,
    kappa: float = 0.01,
    ):
    mu, std = predict(gp_params, xs, ys, mask, xt=x_pred)
    return mu + kappa * std, (mu, std)


def lower_confidence_bounds(
    x_pred: jnp.ndarray,
    xs: jax.Array,
    ys: jax.Array,
    mask: jax.Array,
    gp_params: GPParams,
    kappa: float = 2.576,
    ):
    mu, std = predict(gp_params, xs, ys, mask, xt=x_pred)
    return mu - kappa * std, (mu, std)
