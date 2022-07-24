from functools import partial
from typing import Callable, NamedTuple, Tuple, Union

from jax import jacrev, jit, lax, random, vmap
from jax.tree_util import tree_map
from jax.scipy.optimize import minimize
import jax.numpy as jnp

from bayex.acq import ACQ, select_acq
from bayex.gp import DataTypes, GPParams, GPState, posterior_fit, round_integers
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


def _replace_nan_values(arr: Array) -> Array:
    # todo(alonfnt): Find a more robust solution.
    return jnp.where(jnp.isnan(arr), 0, arr)


def sample(
    key: Array,
    params: GPParams,
    x: Array,
    y: Array,
    bounds: Array,
    dtypes: DataTypes,
    acq: Callable,
    n_seed: int = 100,
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
    bounds: Array of (dim, 2) shape with the lower and upper bounds of the
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

    domain = random.uniform(key1, shape=(n_seed, dim), minval=bounds[:, 0], maxval=bounds[:, 1])

    _acq1 = partial(acq, params=params, x=x, y=y, dtypes=dtypes)
    _acq = lambda x: _acq1(x.reshape(-1, dim)).reshape()

#    J = jacobian(lambda x: _acq(x.reshape(-1, dim)).reshape())
#    HS = vmap(lambda x: x + lr * J(x))
#    domain = lax.fori_loop(0, n_epochs, lambda _, d: HS(d), domain)
    res = vmap(partial(minimize, method='BFGS'), in_axes=(None, 0))(_acq, domain)
    domain = _replace_nan_values(res.x)
    domain = jnp.clip(domain.reshape(-1, dim), a_min=bounds[:, 0], a_max=bounds[:, 1])
    domain = round_integers(domain, dtypes)

    next_X = domain[res.fun.argmax()]
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
    dtypes: dict = None,
    acq: ACQ = ACQ.EI,
    **acq_params: dict,
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

    if dtypes is not None:
        _sorted_types = {k: dtypes[k] for k in _vars if k in dtypes}

    if ctypes is not None:
        _sorted_types = {k: ctypes[k] for k in _vars if k in ctypes}
        dtypes = DataTypes(integers=[_vars.index(k) for k, v in _sorted_types.items() if v == int])
    else:
        dtypes = DataTypes(integers=[])

    _acq = select_acq(acq, acq_params)

    bounds = jnp.asarray(list(_sorted_constrains.values()))

    X = random.uniform(
        key,
        shape=(n_init, dim),
        minval=bounds[:, 0],
        maxval=bounds[:, 1],
    )
    X = round_integers(X, dtypes)
    Y = vmap(f)(*X.T)

    # Expand the array with the same last values to not perjudicate the gp.
    # the reason to apply it as a function is to avoid having twice the memory
    # usage, since JAX does not do inplace updates except after being
    # compiled.
    X = _extend_array(X, n, 0)
    Y = _extend_array(Y, n, 0)

    # Initialize the GP parameters
    params = GPParams(
        noise=jnp.zeros((1, 1)) - 5.0,
        amplitude=jnp.zeros((1, 1)),
        lengthscale=jnp.zeros((1, dim)),
    )
    momentums = tree_map(lambda x: x * 0, params)
    scales = tree_map(lambda x: x * 0 + 1, params)

    for idx in range(n_init, n + n_init):
        params, momentums, scales = posterior_fit(X, Y, params, momentums, scales, dtypes)
        max_params, key = sample(key, params, X, Y, bounds, dtypes, _acq)
        X = X.at[idx, ...].set(max_params)  # type: ignore
        Y = Y.at[idx].set(f(*max_params))  # type: ignore

    best_target = float(Y.max())
    best_params = {k: v for (k, v) in zip(constrains.keys(), X[Y.argmax()])}
    optimizer_params = OptimizerParameters(
        target=best_target, params=best_params, f=f, params_all=X, target_all=Y
    )
    return optimizer_params


class Observable(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray


class MaskedObservable(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    n: int


def add_observable(observables, new_observable) -> MaskedObservable:
    n = observables.n
    current_obs = Observable(observables.x, observables.y)
    new_obs = tree_map(lambda x, y: x.at[n].set(y), current_obs, new_observable)
    return MaskedObservable(new_obs.x, new_obs.y, n=n + 1)


class BayesianOptimizer(NamedTuple):
    init: Callable
    update: Callable
    sample: Callable


def _sample_inputs(key, shape, minval, maxval, dtypes):
    X = random.uniform(key, shape, minval=minval, maxval=maxval)
    X = round_integers(X, dtypes)
    return X


def optimizer(
    fn: Callable, bounds: dict, dtypes: Union[dict, None] = None, nmax=1000
) -> BayesianOptimizer:

    ndim = len(bounds)

    _vars = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    _sorted_bounds = {k: bounds[k] for k in _vars}

    if dtypes is not None:
        _sorted_types = {k: dtypes[k] for k in _vars if k in dtypes}
        dtypes = DataTypes(integers=[vars.index(k) for k, v in _sorted_types.items() if v == int])
    else:
        dtypes = DataTypes(integers=[])

    _bounds = jnp.asarray(list(_sorted_bounds.values()))
    input_sampler = partial(
        _sample_inputs, minval=_bounds[:, 0], maxval=_bounds[:, 1], dtypes=dtypes
    )

    def init_fn(key, num_samples) -> Tuple[GPState, MaskedObservable]:
        params = GPParams(
            noise=jnp.full((1, 1), -5.0),
            amplitude=jnp.zeros((1, 1)),
            lengthscale=jnp.zeros((1, ndim)),
        )
        momentums = tree_map(lambda x: jnp.zeros_like(x), params)
        scales = tree_map(lambda x: jnp.ones_like(x), params)
        gp_state = GPState(params, momentums, scales)

        X = input_sampler(key, (num_samples, ndim))
        Y = vmap(fn)(*jnp.transpose(X))

        X = _extend_array(X, nmax - num_samples, 0)
        Y = _extend_array(Y, nmax - num_samples, 0)
        return gp_state, MaskedObservable(x=X, y=Y, n=num_samples)

    def update_fn(state: GPState, ob: MaskedObservable) -> GPState:
        state = posterior_fit(ob.x, ob.y, state, dtypes=dtypes)
        return state

    _acq = select_acq(ACQ.EI, dict())
    def sample_fn(key, state: GPState, observables: MaskedObservable) -> Observable:
        X, key = sample(key, state.params, observables.x, observables.y, _bounds, dtypes, _acq)
        y = fn(*X)
        return Observable(x=X, y=y)

    return BayesianOptimizer(init=init_fn, update=update_fn, sample=sample_fn)
