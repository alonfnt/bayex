# Based on the Gaussian regression example in
# https://github.com/google/jax/blob/main/examples/gaussian_process_regression.py
from collections import namedtuple
from functools import partial
from typing import Any, Callable, Tuple

import jax.numpy as jnp
import jax.scipy as scipy
from jax import grad, jit, lax, ops, tree_map, tree_multimap, vmap

from bayex.types import Array

GParameters = namedtuple("GParameters", ["noise", "amplitude", "lengthscale"])
DataTypes = namedtuple("DataTypes", ["integers"])


def cov_map(cov_func: Callable, xs: Array, xs2: Array = None) -> Array:
    """
    Computes the covariance matrix of the given function with the
    given points.
    """
    if xs2 is None:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs)
    else:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs2).T


def softplus(x: Array) -> Array:
    return jnp.logaddexp(x, 0.0)


def exp_quadratic(x1: Array, x2: Array, ls: Array) -> Array:
    return jnp.exp(-jnp.sum((x1 - x2) ** 2 / ls ** 2))


def round_vars(arr: Array, indexes: list) -> Array:
    """
    The input variables corresponding to an integer-valued input variable are
    rounded to the closest integer value.
    """
    for idx in indexes:
        arr = ops.index_update(arr, ops.index[:, idx], jnp.round(arr[:, idx]))
    return arr


def gp(
    params: GParameters,
    x: Array,
    y: Array,
    dtypes: DataTypes,
    xt: Array = None,
    compute_ml: bool = False,
) -> Any:
    """
    Gaussian Processor Main function.
    It is used as the based for the trainig of the GP, as well
    as the computing of marginal likelihood and prediction.

    Parameters:
    -----------
    params: Hyperparameters of the GP.
    x: Sampled points.
    y: Target values of the sampled points.
    xt: Points on which to predict.
    compute_ml: Flag indicating whether return the marginal
                distribution or continue with the predictions.

    Returns:
    --------
    The mean and standard deviations of the resulting
    Gaussian Distributions.
    """
    n = x.shape[0]
    x = round_vars(x, dtypes.integers)
    noise, amp, ls = tree_map(softplus, params)
    kernel = partial(exp_quadratic, ls=ls)

    ymean = jnp.mean(y)
    y = y - ymean
    train_cov = amp * cov_map(kernel, x) + jnp.eye(n) * (noise + 1e-6)
    chol = scipy.linalg.cholesky(train_cov, lower=True)
    kinvy = scipy.linalg.solve_triangular(
        chol.T, scipy.linalg.solve_triangular(chol, y, lower=True)
    )
    if compute_ml:
        log2pi = jnp.log(2.0 * jnp.pi)
        ml = jnp.sum(
            -0.5 * jnp.dot(y.T, kinvy)
            - jnp.sum(jnp.log(jnp.diag(chol)))
            - (n / 2.0) * log2pi
        )
        ml -= jnp.sum(-0.5 * jnp.log(2 * 3.1415) - jnp.log(amp) ** 2)
        return -ml

    if xt is not None:
        xt = round_vars(xt, dtypes.integers)

    cross_cov = amp * cov_map(kernel, x, xt)
    mu = jnp.dot(cross_cov.T, kinvy) + ymean
    v = scipy.linalg.solve_triangular(chol, cross_cov, lower=True)
    var = amp * cov_map(kernel, xt) - jnp.dot(v.T, v)

    if n > 1:
        var = jnp.diag(var)

    std = jnp.sqrt(var)
    return mu, std


marginal_likelihood = partial(gp, compute_ml=True)
predict = jit(partial(gp, compute_ml=False))
grad_fun = jit(grad(marginal_likelihood))


@jit
def train(
    x: Array,
    y: Array,
    params: GParameters,
    momentums: GParameters,
    scales: GParameters,
    dtypes: DataTypes,
    lr: float = 0.01,
    nsteps: int = 20,
) -> Tuple[GParameters, GParameters, GParameters]:
    """
    Training function of the Gaussian Process Regressor.

    Parameters:
    -----------
    x: Sampled points.
    y: Target values of the sampled points.
    params: Hyperparameters of the GP.
    lr: Learning rate of the train step.
    nsteps: Number of epochs to train.

    Returns:
    --------
    Tuple with the trained `params`, `momentums` and `scales`.
    """

    def train_step(
        params: GParameters, momentums: GParameters, scales: GParameters
    ) -> Tuple:
        grads = grad_fun(params, x, y, dtypes=dtypes)
        momentums = tree_multimap(
            lambda m, g: 0.9 * m + 0.1 * g, momentums, grads
        )
        scales = tree_multimap(
            lambda s, g: 0.9 * s + 0.1 * g ** 2, scales, grads
        )
        params = tree_multimap(
            lambda p, m, s: p - lr * m / jnp.sqrt(s + 1e-5),
            params,
            momentums,
            scales,
        )
        return params, momentums, scales

    params, momentums, scales = lax.fori_loop(
        0,
        nsteps,
        lambda _, v: train_step(*v),
        (params, momentums, scales),
    )

    return params, momentums, scales
