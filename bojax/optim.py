from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import jax.numpy as jnp
from jax import random, vmap
from jax.tree_util import tree_map

from bojax.acq import expected_improvement
from bojax.gp import GPParams, GPState, posterior_fit
from bojax.observables import DataTypes, Measures, round_integers


class BayesianOptimizer(NamedTuple):
    """Holds the fu"""

    init: Callable
    fit: Callable
    sample: Callable


def _sample_inputs(key, shape, minval, maxval, dtypes):
    X = random.uniform(key, shape, minval=minval, maxval=maxval)
    X = round_integers(X, dtypes)
    return X


def optimizer(
    fn: Callable,
    bounds: dict,
    dtypes: Optional[dict] = None,
    acq: Callable = expected_improvement,
) -> BayesianOptimizer:

    ndim = len(bounds)

    _vars = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    _sorted_bounds = {k: bounds[k] for k in _vars}

    if dtypes is not None:
        _sorted_types = {k: dtypes[k] for k in _vars if k in dtypes}
        _dtypes = DataTypes(integers=[_vars.index(k) for k, v in _sorted_types.items() if v == int])
    else:
        _dtypes = DataTypes(integers=[])

    _bounds = jnp.asarray(list(_sorted_bounds.values()))
    lbounds, ubounds = _bounds[:, 0], _bounds[:, 1]
    input_sampler = partial(_sample_inputs, minval=lbounds, maxval=ubounds, dtypes=_dtypes)

    def init_fn(key, init_samples: int) -> Tuple[GPState, Measures]:
        params = GPParams(
            noise=jnp.full((1, 1), -5.0),
            amplitude=jnp.zeros((1, 1)),
            lengthscale=jnp.zeros((1, ndim)),
        )
        momentums = tree_map(jnp.zeros_like, params)
        scales = tree_map(jnp.ones_like, params)
        gp_state = GPState(params, momentums, scales)

        X = input_sampler(key, (init_samples, ndim))
        Y = vmap(fn)(*jnp.transpose(X))
        return gp_state, Measures(x=X, y=Y)

    def fit_fn(state: GPState, ob: Measures) -> GPState:
        state = posterior_fit(ob.x, ob.y, state, dtypes=_dtypes)
        return state

    def sample_fn(key, state: GPState, observables: Measures, n_seed=1000) -> Measures:
        domain = input_sampler(key, shape=(n_seed, ndim))
        x, y = observables
        results = acq(domain, state.params, x, y, dtypes=_dtypes)
        X = domain[jnp.argmax(results)]
        y = fn(*X)
        return Measures(x=X, y=y)

    return BayesianOptimizer(init=init_fn, fit=fit_fn, sample=sample_fn)
