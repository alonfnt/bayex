from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
from jax import jacrev, jit, lax, ops, partial, random, vmap, tree_map
from jax.scipy.stats import norm

from .gp import GParameters, predict, train


Array = Any


def jacobian(f: Callable) -> Callable:
    return jit(jacrev(f))


def expected_improvement(
    x_pred: Array,
    params: GParameters,
    x: Array,
    y: Array,
    xi: float = 0.01,
) -> Array:
    """
    Computes the expected improvement at points x_pred over a
    Gaussian process trained on x and y.

    Parameters:
    -----------
    x_pred: The points at which the improvement is computed.
    params: Trained hyperparameters of the GP.
    x: Sampled points.
    y: Target values of the sampled points.
    xi: Parameter to balance exploration-exploitation.

    Returns:
    --------
    ei: The expected improvement of x_pred on the trained GP.
    """
    y_max = y.max()
    mu, std = predict(params, x, y, xt=x_pred)
    improvement = mu.T - y_max - xi
    z = improvement / (std + 1e-3)
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    return ei


def replace_nan_values(arr: Array) -> Array:
    """
    Replaces the NaN values (if any) in arr with 0.

    Parameters:
    -----------
    arr: The array where NaN is removed from.

    Returns:
    --------
    The array with all the NaN elements replaced with 0.
    """
    # todo(alonfnt): Find a more robust solution.
    return jnp.where(jnp.isnan(arr), 0, arr)


@jit
def suggest_next(
    key: Array,
    params: GParameters,
    x: Array,
    y: Array,
    bounds: Array,
    n_seed: int = 1000,
    lr: float = 0.1,
    n_epochs: int = 150,
) -> Tuple[Array, Array]:
    """
    Suggests the new point to sample by optimizing the acquisition function.

    Parameters:
    -----------
    key: The pseudo-random generator key used for jax random functions.
    params: Hyperparameters of the Gaussian Process Regressor.
    x: Sampled points.
    y: Sampled targets.
    bounds: Array of (2, dim) shape with the lower and upper bounds of the
            variables.y_max: The current maximum value of the target values Y.
    n_seed (optional): the number of points to probe and minimize until
            finding the one that maximizes the acquisition functions.
    lr (optional): The step size of the gradient descent.
    n_epochs (optional): The number of steps done on the descent to minimize
            the seeds.


    Returns:
    --------
    A tuple with the parameters that maximize the acquisition function and a
    jax PRGKey to be used in the next sampling.
    """

    key1, key2 = random.split(key, 2)
    dim = bounds.shape[0]

    domain = random.uniform(
        key1, shape=(n_seed, dim), minval=bounds[:, 0], maxval=bounds[:, 1]
    )

    acq = partial(expected_improvement, params=params, x=x, y=y)

    J = jacobian(lambda x: acq(x.reshape(-1, dim)).reshape())
    HS = vmap(lambda x: x + lr * J(x))

    domain = lax.fori_loop(0, n_epochs, lambda _, d: HS(d), domain)
    domain = jnp.clip(
        domain.reshape(-1, dim), a_min=bounds[:, 0], a_max=bounds[:, 1]
    )
    domain = replace_nan_values(domain)

    ys = acq(domain)
    next_X = domain[ys.argmax()]
    return next_X, key2


def optim(
    f: Callable,
    constrains: Dict,
    seed: int = 42,
    n_init: int = 5,
    n: int = 10,
    xi: float = 0.01,
):
    """
    Finds the inputs of 'f' that yield the maximum value between the given
    'constrains', after 'n_init' + 'n' iterations.

    Parameters:
    -----------
    f: Function to optimize.
    constrains: Dictionary with the domain of each input variable.
    seed: Pseudo-random number generator seed for reproducibility
    n_init: Number of initial evaluations before suggesting optimized samples.
    n: Number of sampling iterations.
    xi: Parameter to balance exploration-exploitation.

    Returns:
    --------
    The parameters that maximize the given 'f'.
    """

    assert n > 0, "Num of iterations n should be a positive integer"

    key = random.PRNGKey(seed)
    dim = len(constrains)
    bounds = jnp.asarray(list(constrains.values()))
    X = random.uniform(
        key,
        shape=(n_init, dim),
        minval=bounds[:, 0],
        maxval=bounds[:, 1],
    )
    Y = vmap(f)(*X.T)

    # Create an empty array with the same values to not perjudicate the gp
    X = jnp.pad(X, ((0, n), (0, 0)), mode="edge")
    Y = jnp.pad(Y, ((0, n)), mode="edge")

    # Initialize the GP parameters
    params = GParameters(
        noise=jnp.zeros((1, 1)) - 5.0,
        amplitude=jnp.zeros((1, 1)),
        lengthscale=jnp.zeros((1, 1)),
    )
    momentums = tree_map(lambda x: x * 0, params)
    scales = tree_map(lambda x: x * 0 + 1, params)

    # todo(alonfnt): make it JAX friendly
    for i in range(n):
        idx = n_init + i
        params, momentums, scales = train(
            x=X, y=Y, params=params, momentums=momentums, scales=scales
        )
        max_params, key = suggest_next(
            key,
            params,
            X,
            Y,
            bounds,
        )
        print(f"New max point: {max_params} and {f(*max_params)}")
        X = ops.index_update(X, ops.index[idx, ...], max_params)
        Y = ops.index_update(Y, ops.index[idx], f(*max_params))
    xmax = X[Y.argmax()]
    return xmax
