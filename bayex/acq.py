from enum import Enum
from functools import partial
from typing import Callable, Union

from jax.scipy.stats import norm

from bayex.gp import GParameters, predict
from bayex.types import Array


class ACQ(Enum):
    EI = 1
    POI = 2
    UCB = 3
    LCB = 4


def select_acq(acq: Union[ACQ, str], acq_params: dict) -> Callable:
    """
    Wrapper that selects the correct acquisition function and makes sure
    that the parameters for it exist.
    """
    # note(alonfnt): 3.10 asks for a switch statement here, but for <3.10
    # if/else makes more sense.

    if acq == ACQ.EI:
        xi = acq_params["xi"] if "xi" in acq_params else 0.01
        return partial(expected_improvement, xi=xi)
    elif acq == ACQ.POI:
        xi = acq_params["xi"] if "xi" in acq_params else 0.01
        return partial(probability_improvement, xi=xi)
    elif acq == ACQ.UCB:
        kappa = acq_params["kappa"] if "kappa" in acq_params else 0.01
        return partial(upper_confidence_bounds, kappa=kappa)
    elif acq == ACQ.LCB:
        kappa = acq_params["kappa"] if "kappa" in acq_params else 0.01
        return partial(lower_confidence_bounds, kappa=kappa)
    raise ValueError("The acquisition function given is not correct.")


def expected_improvement(
    x_pred: Array,
    params: GParameters,
    x: Array,
    y: Array,
    dtypes: Union[dict, None],
    xi: dict,
) -> Array:
    y_max = y.max()
    mu, std = predict(params, x, y, dtypes, xt=x_pred)
    improvement = mu.T - y_max - xi
    z = improvement / (std + 1e-3)
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    return ei


def probability_improvement(
    x_pred: Array,
    params: GParameters,
    x: Array,
    y: Array,
    dtypes: Union[dict, None],
    xi: float,
) -> Array:
    y_max = y.max()
    mu, std = predict(params, x, y, dtypes, xt=x_pred)
    z = (mu - y_max - xi) / std
    return norm.cdf(z)


def upper_confidence_bounds(
    x_pred: Array,
    params: GParameters,
    x: Array,
    y: Array,
    dtypes: Union[dict, None],
    kappa: float,
) -> Array:
    mu, std = predict(params, x, y, dtypes, xt=x_pred)
    return mu + kappa * std


def lower_confidence_bounds(
    x_pred: Array,
    params: GParameters,
    x: Array,
    y: Array,
    dtypes: Union[dict, None],
    kappa: float,
) -> Array:
    mu, std = predict(params, x, y, dtypes, xt=x_pred)
    return mu - kappa * std
