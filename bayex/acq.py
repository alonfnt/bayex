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
    r"""
    Expected Improvement (EI) acquisition function.

    Favors points with high improvement over the current best observed value,
    balancing exploitation and exploration.

    The formula is:

    .. math::

        EI(x) = (\mu(x) - y^* - \xi) \Phi(z) + \sigma(x) \phi(z)

    where:

    .. math::

        z = \frac{\mu(x) - y^* - \xi}{\sigma(x)}

    Args:
        x_pred: Candidate input locations to evaluate.
        xs: Observed inputs.
        ys: Observed function values.
        mask: Boolean mask indicating valid entries in `ys`.
        gp_params: Gaussian Process hyperparameters.
        xi: Exploration-exploitation tradeoff parameter.

    Returns:
        EI scores at `x_pred`.
    """
    ymax = jnp.max(ys, where=mask.astype(bool), initial=-jnp.inf)
    mu, std = predict(gp_params, xs, ys, mask, xt=x_pred)
    a = mu - ymax - xi
    z = a / (std + 1e-3)
    ei = a * norm.cdf(z) + std * norm.pdf(z)
    return ei


def probability_improvement(
    x_pred: jnp.ndarray,
    xs: jax.Array,
    ys: jax.Array,
    mask: jax.Array,
    gp_params: GPParams,
    xi: float = 0.01,
    ):
    r"""
    Probability of Improvement (PI) acquisition function.

    Estimates the probability that a candidate point will improve
    over the current best observed value.

    The formula is:

    .. math::

        PI(x) = \Phi\left(\frac{\mu(x) - y^* - \xi}{\sigma(x)}\right)

    Args:
        x_pred: Candidate input locations to evaluate.
        xs: Observed inputs.
        ys: Observed function values.
        mask: Boolean mask indicating valid entries in `ys`.
        gp_params: Gaussian Process hyperparameters.
        xi: Improvement margin for sensitivity.

    Returns:
        PI scores at `x_pred`.
    """
    y_max = ys.max()
    mu, std = predict(gp_params, xs, ys, mask, xt=x_pred)
    z = (mu - y_max - xi) / std
    return norm.cdf(z)


def upper_confidence_bounds(
    x_pred: jnp.ndarray,
    xs: jax.Array,
    ys: jax.Array,
    mask: jax.Array,
    gp_params: GPParams,
    kappa: float = 0.01,
    ):
    r"""
    Upper Confidence Bound (UCB) acquisition function.

    Promotes exploration by favoring points with high predictive uncertainty.

    The formula is:

    .. math::

        UCB(x) = \mu(x) + \kappa \cdot \sigma(x)

    Args:
        x_pred: Candidate input locations to evaluate.
        xs: Observed inputs.
        ys: Observed function values.
        mask: Boolean mask indicating valid entries in `ys`.
        gp_params: Gaussian Process hyperparameters.
        kappa: Weighting factor for uncertainty.

    Returns:
        UCB scores at `x_pred`.
    """
    mu, std = predict(gp_params, xs, ys, mask, xt=x_pred)
    return mu + kappa * std


def lower_confidence_bounds(
    x_pred: jnp.ndarray,
    xs: jax.Array,
    ys: jax.Array,
    mask: jax.Array,
    gp_params: GPParams,
    kappa: float = 2.576,
    ):
    r"""
    Lower Confidence Bound (LCB) acquisition function.

    Useful for minimization tasks. Encourages sampling in uncertain regions
    with low predicted values.

    The formula is:

    .. math::

        LCB(x) = \mu(x) - \kappa \cdot \sigma(x)

    Args:
        x_pred: Candidate input locations to evaluate.
        xs: Observed inputs.
        ys: Observed function values.
        mask: Boolean mask indicating valid entries in `ys`.
        gp_params: Gaussian Process hyperparameters.
        kappa: Weighting factor for uncertainty.

    Returns:
        LCB scores at `x_pred`.
    """
    mu, std = predict(gp_params, xs, ys, mask, xt=x_pred)
    return mu - kappa * std
