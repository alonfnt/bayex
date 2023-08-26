from collections import namedtuple
from functools import partial
from typing import Any, Callable, Tuple, Union

from jax import grad, jit, lax, ops, tree_util, vmap
import jax.numpy as jnp
from jax.scipy.linalg import cholesky, solve_triangular

from bayex.types import Array

GParameters = namedtuple("GParameters", ["noise", "amplitude", "lengthscale"])
DataTypes = namedtuple("DataTypes", ["integers"])


def cov(k: Callable, x1: Array, x2: Array = None) -> Array:
    """
    Computes the covariance matrix of the given kernel with the given points.
    """
    if x2 is None:
        return vmap(lambda x: vmap(lambda y: k(x, y))(x1))(x1)
    else:
        return vmap(lambda x: vmap(lambda y: k(x, y))(x1))(x2).T


def softplus(x: Array) -> Array:
    return jnp.logaddexp(x, 0.0)


def exp_quadratic(x1: Array, x2: Array, ls: Array) -> Array:
    return jnp.exp(-jnp.sum((x1 - x2) ** 2 / ls ** 2))


def round_integers(arr: Array, dtypes: Union[DataTypes, None]) -> Array:
    """
    The input variables corresponding to an integer-valued input variable are
    rounded to the closest integer value.
    """
    if dtypes is None:
        return arr

    indexes = dtypes.integers
    for idx in indexes:
        arr = ops.index_update(arr, ops.index[:, idx], jnp.round(arr[:, idx]))
    return arr


def gaussian_process(
    params: GParameters,
    x: Array,
    y: Array,
    dtypes: DataTypes = None,
    xt: Array = None,
    compute_ml: bool = False,
) -> Any:
    """
    Base function that deals with the Gaussian Processes.
    """
    # Number of points in the prior distribution
    n = x.shape[0]

    # Rounding integer values before computing the covariance matrices.
    x = round_integers(x, dtypes)

    noise, amp, ls = tree_util.tree_map(softplus, params)
    kernel = partial(exp_quadratic, ls=ls)

    # Normalization of measurements
    ymean = jnp.mean(y)
    y = y - ymean

    # Covariance matrix K[X,X] with noise measurements
    K = amp * cov(kernel, x) + (jnp.eye(n) * (noise + 1e-6))

    # In order to compute the inverse of K, i.e K^-1, we make use of Cholesky
    # factorization K = LxL^T to improve performance on the solving.
    L = cholesky(K, lower=True)
    K_inv_y = solve_triangular(L.T, solve_triangular(L, y, lower=True))

    if compute_ml:
        # Compute the marginal likelihood using its closed form:
        # log(P) = - 0.5 yK^-1y - 0.5 |K-sigmaI| - n/2 log(2pi)
        fitting = y.T.dot(K_inv_y)

        # Compute the determinant using the Lower Diagonal Factorization
        # since it makes it only the diagonal multiplication (sum of logs)
        penalty = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

        ml = -0.5 * jnp.sum(fitting + penalty + n * jnp.log(2.0 * jnp.pi))

        # Add the amplitude hyperparameter to the marginal likelihood
        ml += 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(amp.reshape()) ** 2
        return -ml

    if xt is not None:
        xt = round_integers(xt, dtypes)

    # Compute the covariance with the new point xt
    K_cross = amp * cov(kernel, x, xt)

    # Return the mean and standard devition of the Gaussian Proceses
    mean = jnp.dot(K_cross.T, K_inv_y) + ymean
    v = solve_triangular(L, K_cross, lower=True)
    var = amp * cov(kernel, xt) - jnp.dot(v.T, v)
    std = jnp.sqrt(var) if n == 1 else jnp.diag(var)
    return mean, std


marginal_likelihood = partial(gaussian_process, compute_ml=True)
grad_fun = jit(grad(marginal_likelihood))
predict = jit(partial(gaussian_process, compute_ml=False))


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
        momentums = tree_util.tree_map(
            lambda m, g: 0.9 * m + 0.1 * g, momentums, grads
        )
        scales = tree_util.tree_map(
            lambda s, g: 0.9 * s + 0.1 * g ** 2, scales, grads
        )
        params = tree_util.tree_map(
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
