from typing import Union

import jax.numpy as jnp
from jax.scipy.stats import norm

from bayex.gp import GPParams, predict
from bayex.observables import DataTypes


def expected_improvement(
    x_pred: jnp.ndarray,
    params: GPParams,
    x: jnp.ndarray,
    y: jnp.ndarray,
    dtypes: DataTypes,
    xi: float = 0.01,
) -> jnp.ndarray:
    y_max = y.max()
    mu, std = predict(params, x, y, dtypes, xt=x_pred)
    improvement = mu.T - y_max - xi
    z = improvement / (std + 1e-3)
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    return ei


def probability_improvement(
    x_pred: jnp.ndarray,
    params: GPParams,
    x: jnp.ndarray,
    y: jnp.ndarray,
    dtypes: DataTypes,
    xi: float,
) -> jnp.ndarray:
    y_max = y.max()
    mu, std = predict(params, x, y, dtypes, xt=x_pred)
    z = (mu - y_max - xi) / std
    return norm.cdf(z)


def upper_confidence_bounds(
    x_pred: jnp.ndarray,
    params: GPParams,
    x: jnp.ndarray,
    y: jnp.ndarray,
    dtypes: DataTypes,
    kappa: float,
) -> jnp.ndarray:
    mu, std = predict(params, x, y, dtypes, xt=x_pred)
    return mu + kappa * std


def lower_confidence_bounds(
    x_pred: jnp.ndarray,
    params: GPParams,
    x: jnp.ndarray,
    y: jnp.ndarray,
    dtypes: Union[dict, None],
    kappa: float,
) -> jnp.ndarray:
    mu, std = predict(params, x, y, dtypes, xt=x_pred)
    return mu - kappa * std
