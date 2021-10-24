from typing import Callable, NamedTuple, Tuple, Union

import jax.numpy as jnp
from jax import jacrev, jit, lax, ops, random, tree_map, vmap
from jax.scipy.stats import norm
from functools import partial

from bayex.gp import DataTypes, GParameters, predict, round_vars, train
from bayex.types import Array


class OptimizerParameters(NamedTuple):
    """
    Object holding the results of the optimization.
    """

    target: Union[Array, float]
    params: Array
    f: Callable
    params_all: Array
    target_all: Array


def jacobian(f: Callable) -> Callable:
    return jit(jacrev(f))


def expected_improvement(
    x_pred: Array,
    params: GParameters,
    x: Array,
    y: Array,
    xi: float = 0.01,
    dtypes: Union[dict, None] = None,
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
    mu, std = predict(params, x, y, dtypes, xt=x_pred)
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
    dtypes: DataTypes,
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
    dtypes: The type of non-real variables in the target function.
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

    acq = partial(expected_improvement, params=params, x=x, y=y, dtypes=dtypes)

    J = jacobian(lambda x: acq(x.reshape(-1, dim)).reshape())
    HS = vmap(lambda x: x + lr * J(x))

    domain = lax.fori_loop(0, n_epochs, lambda _, d: HS(d), domain)
    domain = jnp.clip(
        domain.reshape(-1, dim), a_min=bounds[:, 0], a_max=bounds[:, 1]
    )
    domain = replace_nan_values(domain)
    domain = round_vars(domain, dtypes.integers)

    ys = acq(domain)
    next_X = domain[ys.argmax()]
    return next_X, key2


@partial(jit, static_argnums=(1, 2))
def _extend_array(arr: Array, pad_width: int, axis: int) -> Array:
    """
    Extends the array pad_width only on one direction and fills it with
    the last value of that axis.
    TODO: consider donate_argnums=0 if the device allows it.
    """
    pad_shape = [(0, 0)] * arr.ndim
    pad_shape[axis] = (0, pad_width)
    return jnp.pad(arr, pad_shape, mode="edge")


def optim(
    f: Callable,
    constrains: dict,
    seed: int = 42,
    n_init: int = 5,
    n: int = 10,
    xi: float = 0.01,
    ctypes: dict = None,
) -> OptimizerParameters:
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
    ctypes: The type of non-real variables in the target function.

    Returns:
    --------
    The parameters that maximize the given 'f'.
    """

    assert n > 0, "Num of iterations n should be a positive integer"

    key = random.PRNGKey(seed)
    dim = len(constrains)
    _vars = f.__code__.co_varnames[: f.__code__.co_argcount]
    _sorted_constrains = {k: constrains[k] for k in _vars}

    if ctypes is not None:
        _sorted_types = {k: ctypes[k] for k in _vars if k in ctypes}
        dtypes = DataTypes(
            integers=[
                _vars.index(k) for k, v in _sorted_types.items() if v == int
            ]
        )
    else:
        dtypes = DataTypes(integers=[])

    bounds = jnp.asarray(list(_sorted_constrains.values()))

    X = random.uniform(
        key,
        shape=(n_init, dim),
        minval=bounds[:, 0],
        maxval=bounds[:, 1],
    )
    X = round_vars(X, dtypes.integers)
    Y = vmap(f)(*X.T)

    # Expand the array with the same last values to not perjudicate the gp.
    # the reason to apply it as a function is to avoid having twice the memory
    # usage, since JAX does not do inplace updates except after being
    # compiled.
    X = _extend_array(X, n, 0)
    Y = _extend_array(Y, n, 0)

    # Initialize the GP parameters
    params = GParameters(
        noise=jnp.zeros((1, 1)) - 5.0,
        amplitude=jnp.zeros((1, 1)),
        lengthscale=jnp.zeros((1, dim)),
    )
    momentums = tree_map(lambda x: x * 0, params)
    scales = tree_map(lambda x: x * 0 + 1, params)

    # todo(alonfnt): make it JAX friendly
    for i in range(n):
        idx = n_init + i
        params, momentums, scales = train(
            X, Y, params, momentums, scales, dtypes
        )
        max_params, key = suggest_next(key, params, X, Y, bounds, dtypes)
        X = ops.index_update(X, ops.index[idx, ...], max_params)
        Y = ops.index_update(Y, ops.index[idx], f(*max_params))

    best_target = float(Y.max())
    best_params = {k: v for (k, v) in zip(constrains.keys(), X[Y.argmax()])}
    optimizer_params = OptimizerParameters(
        target=best_target, params=best_params, f=f, params_all=X, target_all=Y
    )
    return optimizer_params
