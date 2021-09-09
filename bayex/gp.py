# Based on the Gaussian regression example in
# https://github.com/google/jax/blob/main/examples/gaussian_process_regression.py
from collections import namedtuple
from functools import partial
from typing import Any, Callable, Tuple

import jax.numpy as jnp
import jax.scipy as scipy
from jax import grad, jit, tree_map, tree_multimap, vmap, lax

Array = Any  # waiting for JAX official type support

GParameters = namedtuple(
    "GaussianParameters", ["noise", "amplitude", "lengthscale"]
)


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


def exp_quadratic(x1: Array, x2: Array) -> Array:
    return jnp.exp(-jnp.sum((x1 - x2) ** 2))


def gp(params, x, y, xt=None, compute_ml=False):
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

    noise, amp, ls = tree_map(softplus, params)

    ymean = jnp.mean(y)
    y = y - ymean
    x = x / ls
    train_cov = amp * cov_map(exp_quadratic, x) + jnp.eye(n) * (noise + 1e-6)
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
        xt = xt / ls

    cross_cov = amp * cov_map(exp_quadratic, x, xt)
    mu = jnp.dot(cross_cov.T, kinvy) + ymean
    v = scipy.linalg.solve_triangular(chol, cross_cov, lower=True)
    var = amp * cov_map(exp_quadratic, xt) - jnp.dot(v.T, v)

    if n > 1:
        var = jnp.diag(var)

    std = jnp.sqrt(var)
    return mu, std


marginal_likelihood = partial(gp, compute_ml=True)
predict = jit(partial(gp, compute_ml=False))
grad_fun = jit(grad(marginal_likelihood))


def train_step(
    params: GParameters,
    momentums: GParameters,
    scales: GParameters,
    x: Array,
    y: Array,
    lr: float = 0.01,
) -> Tuple[GParameters, GParameters, GParameters]:
    """
    Training step of the Gaussian Process Regressor.
    """
    grads = grad_fun(params, x, y)
    momentums = tree_multimap(lambda m, g: 0.9 * m + 0.1 * g, momentums, grads)
    scales = tree_multimap(lambda s, g: 0.9 * s + 0.1 * g ** 2, scales, grads)
    params = tree_multimap(
        lambda p, m, s: p - lr * m / jnp.sqrt(s + 1e-5),
        params,
        momentums,
        scales,
    )
    return params, momentums, scales


@jit
def train(
    x: Array,
    y: Array,
    params: GParameters,
    momentums: GParameters,
    scales: GParameters,
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

    params, momentums, scales = lax.fori_loop(
        0,
        nsteps,
        lambda _, v: train_step(v[0], v[1], v[2], x, y, lr),
        (params, momentums, scales),
    )

    return params, momentums, scales
