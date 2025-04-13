import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from bayex.gp import GPParams, predict


def expected_improvement(
    x_pred: jnp.ndarray,
    xs: jax.Array,
    ys: jax.Array,
    mask: jax.Array,
    gp_params: GPParams,
    xi: float = 0.01,
    ):
    """
    Expected Improvement (EI) acquisition function.

    Favors points with high improvement over the current best observed value,
    balancing exploitation and exploration.

    Args:
        x_pred: Candidate input locations to evaluate.
        xs: Observed inputs.
        ys: Observed function values.
        mask: Boolean mask indicating valid entries in `ys`.
        gp_params: Gaussian Process hyperparameters.
        xi: Exploration-exploitation tradeoff parameter.

    Returns:
        Tuple of:
            - EI scores at `x_pred`.
            - Tuple of (mu, std) from GP prediction.
    """
    ymax = jnp.max(ys, where=mask.astype(bool), initial=-jnp.inf)
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
    """
    Probability of Improvement (PI) acquisition function.

    Estimates the probability that a candidate point will improve
    over the current best observed value.

    Args:
        x_pred: Candidate input locations to evaluate.
        xs: Observed inputs.
        ys: Observed function values.
        mask: Boolean mask indicating valid entries in `ys`.
        gp_params: Gaussian Process hyperparameters.
        xi: Improvement margin for sensitivity.

    Returns:
        Tuple of:
            - PI scores at `x_pred`.
            - Tuple of (mu, std) from GP prediction.
    """
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
    """
    Upper Confidence Bound (UCB) acquisition function.

    Promotes exploration by favoring points with high predictive uncertainty.

    Args:
        x_pred: Candidate input locations to evaluate.
        xs: Observed inputs.
        ys: Observed function values.
        mask: Boolean mask indicating valid entries in `ys`.
        gp_params: Gaussian Process hyperparameters.
        kappa: Weighting factor for uncertainty.

    Returns:
        Tuple of:
            - UCB scores at `x_pred`.
            - Tuple of (mu, std) from GP prediction.
    """
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
    """
    Lower Confidence Bound (LCB) acquisition function.

    Useful for minimization tasks. Encourages sampling in uncertain regions
    with low predicted values.

    Args:
        x_pred: Candidate input locations to evaluate.
        xs: Observed inputs.
        ys: Observed function values.
        mask: Boolean mask indicating valid entries in `ys`.
        gp_params: Gaussian Process hyperparameters.
        kappa: Weighting factor for uncertainty.

    Returns:
        Tuple of:
            - LCB scores at `x_pred`.
            - Tuple of (mu, std) from GP prediction.
    """
    mu, std = predict(gp_params, xs, ys, mask, xt=x_pred)
    return mu - kappa * std, (mu, std)
