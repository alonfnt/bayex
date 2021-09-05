from typing import Callable, Dict, Tuple

import jax.numpy as jnp
from jax import jacrev, jit, lax, ops, partial, random, vmap
from jax._src.prng import PRNGKeyArray
from jax.scipy.stats import norm
from jaxlib.xla_extension import DeviceArray

from .gp import predict, train


def jacobian(f):
    return jit(jacrev(f))


def expected_improvement(x_pred, A, x, y, y_max, xi=0.01):
    """
    Computes the expected improvement at points x_pred over a
    gaussian process trained on x and y.
    """
    mu, std = predict(A, x, y, xtest=x_pred)
    imp = mu.T - y_max - xi
    z = imp / std
    ei = imp * norm.cdf(z) + std * norm.pdf(z)
    return ei


@jit
def suggest_next(
    key: PRNGKeyArray,
    A: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    bounds: jnp.ndarray,
    y_max: float,
    n_seed: int = 10,
    lr: float = 0.1,
    n_epochs: int = 150,
) -> Tuple[DeviceArray, PRNGKeyArray]:
    """
    Suggests the new point to sample by optimizing the acquisition function.

    Parameters:
    -----------
    key: The pseudo-random generator key used for jax random functions.
    acq_fun: The acquisition function to maximize.
    pred_fun: The prediction function with only 1 argument of type Array.
    bounds: Array of (2, dim) shape with the lower and upper bounds of the
            variables.y_max: The current maximum value of the target values Y.
    n_seed (optional): the number of points to probe and minimize until
            finding the one that maximizes the acquisition functions.


    Returns:
    --------

    """

    key1, key2 = random.split(key, 2)
    dim = bounds.shape[0]

    domain = random.uniform(
        key1, shape=(n_seed, dim), minval=bounds[:, 0], maxval=bounds[:, 1]
    )

    acq = partial(expected_improvement, A=A, x=x, y=y, y_max=y_max)

    def minObj(x):
        return -acq(x.reshape(-1, dim)).reshape()

    J = jacobian(minObj)
    HS = vmap(lambda x: x - lr * J(x))

    domain = lax.fori_loop(0, n_epochs, lambda _, d: HS(d), domain)

    clipped_domain = jnp.clip(
        domain.flatten(), a_min=bounds[:, 0], a_max=bounds[:, 1]
    )
    ys = acq(clipped_domain)
    x_max = clipped_domain[ys.argmax()]
    return x_max, key2


def expand(arr, new_size, axis=0):
    shape = arr.shape
    new_shape = ops.index_update(shape, axis, new_size)
    addition = jnp.empty(new_shape)
    new_arr = jnp.concatenate((arr, addition), axis=axis)


def optim(
    f: Callable,
    constrains: Dict,
    seed: int,
    n_init: int = 5,
    n: int = 10,
    xi: float = 0.01,
):

    assert n > 0, "num of iterations n should be a positive integer"

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
    X = jnp.vstack((X, jnp.empty((n, dim))))
    X = ops.index_update(X, ops.index[n_init:], X[n_init - 1])
    Y = jnp.hstack((Y, jnp.empty((n))))
    Y = ops.index_update(Y, ops.index[n_init:], Y[n_init - 1])

    A, p, s = jnp.zeros((1, 1)), 0, 1

    for i in range(n):
        idx = n_init + i
        A, p, s = train(A, p=p, scale=s, x=X, y=Y, nsteps=5)
        xmax, key = suggest_next(
            key,
            A,
            X,
            Y,
            bounds,
            y_max=Y.max(),
        )
        print(f"New Max point: {xmax} and {f(xmax)}")
        X = ops.index_update(X, ops.index[idx, ...], xmax)
        Y = ops.index_update(Y, ops.index[idx], f(xmax))
    xmax = X[Y.argmax()]
    return xmax
