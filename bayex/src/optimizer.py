from functools import partial
from typing import Union, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import bayex.src.acq as boacq
from bayex.src.gp import GPParams, GPState, posterior_fit


class OptimizerState(NamedTuple):
    params: dict
    ys: Union[jax.Array, np.ndarray]
    best_score: float
    best_params: dict
    mask: jax.Array
    gp_state: GPState


class Optimizer:
    """
    A Bayesian optimization class for optimizing expensive-to-evaluate functions.

    Attributes
    ----------
    domain : dict
        A dictionary defining the domain of the parameters to optimize. Each entry specifies the type and domain of a parameter.
    acq : str
        The acquisition function to use. Supported values are 'EI' for Expected Improvement, 'PI' for Probability of Improvement, 'UCB' for Upper Confidence Bound, and 'LCB' for Lower Confidence Bound.
    maximize : bool
        If True, the optimizer seeks to maximize the objective function. If False, it minimizes the function.

    Methods
    -------
    init(self, ys, params):
        Initializes the optimizer state with initial observations and corresponding parameters.

    sample(self, key, opt_state, size=1000, has_prior=False):
        Samples new parameters based on the current optimizer state and acquisition function.

    fit(self, opt_state, y, new_params):
        Updates the optimizer state with a new observation.
    """

    def __init__(self, domain, acq='EI', maximize=False):
        self.domain = domain
        best_fn = jnp.max if maximize else jnp.min
        self.initial = -jnp.inf if maximize else jnp.inf
        self.best_fn = best_fn
        self.best_params_fn = jnp.argmax if maximize else jnp.argmin

        if acq == 'EI':
            self.acq = jax.jit(boacq.expected_improvement)
        elif acq == 'PI':
            self.acq = jax.jit(boacq.probability_improvement)
        elif acq == 'UCB':
            self.acq = jax.jit(boacq.upper_confidence_bounds)
        elif acq == 'LCB':
            self.acq = jax.jit(boacq.lower_confidence_bounds)
        else:
            raise ValueError(f"Acquisition function {acq} is not implemented")

    def init(self, ys, params):
        """
        Initializes the optimizer state with initial observations and corresponding parameters.

        Parameters
        ----------
        ys : Union[jax.Array, np.ndarray]
            The initial set of objective function values corresponding to the initial parameters.
        params : dict
            A dictionary of the initial parameters. Each key should match a key in the domain, and the value should be an array of parameter values.

        Returns
        -------
        OptimizerState
            The initialized state of the optimizer, including the best score and parameters found so far.
        """
        # Create a padded jax array for each parameter and each score.
        # In order to keep jax compilations at a bay.
        num_entries = len(ys)
        pad_value = int(np.ceil(len(ys) / 10) * 10)

        # Convert to jax arrays if they are not already
        ys = jnp.asarray(ys)
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), params)

        # Define padded arrays for the inputs and the outputs
        mask = jnp.zeros(shape=(pad_value,), dtype=jnp.bool_).at[:num_entries].set(True)
        ys = jnp.zeros(shape=(pad_value,), dtype=ys.dtype).at[:num_entries].set(ys)

        _params = {}
        for key, entries in params.items():
            # Assert that the parameter is in the domain dictionary
            assert key in self.domain, f"Parameter {key} is not in the domain"

            # Get dytpe from the domain and create a padded array
            dtype = self.domain[key].dtype
            values = jnp.zeros(shape=(pad_value,), dtype=dtype).at[:num_entries].set(entries)
            _params[key] = values

        # From the given observation, find the better one (either maxima or minima) and return the
        # initial optizer state.
        best_score = float(self.best_fn(ys[mask]))
        best_params_idx = self.best_params_fn(ys[mask])
        best_params = jax.tree_util.tree_map(lambda x: x[mask][best_params_idx], _params)

        # Initialize the gaussian processes state
        gpparams = GPParams(
            noise=jnp.full((1, 1), -5.0),
            amplitude=jnp.zeros((1, 1)),
            lengthscale=jnp.zeros((1, len(_params))),
        )
        momentums = jax.tree_util.tree_map(jnp.zeros_like, gpparams)
        scales = jax.tree_util.tree_map(jnp.ones_like, gpparams)
        gp_state = GPState(gpparams, momentums, scales)

        # Fit to the current observations
        xs = jnp.stack([self.domain[key].transform(_params[key]) for key in _params], axis=1)
        gp_state = posterior_fit(ys, xs, mask=mask, state=gp_state)

        opt_state = OptimizerState(params=_params, ys=ys, best_score=best_score,
                                   best_params=best_params, mask=mask, gp_state=gp_state)

        return opt_state

    def sample(self, key, opt_state, size=1000, has_prior=False):
        """
        Samples new parameters based on the acquisition function and current state of the optimizer.

        Parameters
        ----------
        key : jax.random.PRNGKey
            A PRNGKey used for random number generation in JAX.
        opt_state : OptimizerState
            The current state of the optimizer.
        size : int, optional
            The number of samples to generate. Defaults to 1000.
        has_prior : bool, optional
            If True, includes prior mean and standard deviation in the return values. Defaults to False.

        Returns
        -------
        dict
            A dictionary of sampled parameters that potentially improve the objective function.
        tuple, optional
            A tuple of arrays (means, stds) representing the prior mean and standard deviation of
            the sampled parameters. Only returned if has_prior is True.
        """
        # Sample 'size' elements of each distribution.
        keys = jax.random.split(key, len(opt_state.params))
        samples = {param: self.domain[param].sample(key, (size,))
                   for key, param in zip(keys, opt_state.params)}


        xs = jnp.stack([self.domain[key].transform(opt_state.params[key])
                        for key in opt_state.params], axis=1)
        ys = opt_state.ys
        mask = opt_state.mask
        gpparams = opt_state.gp_state.params
        keys = jax.random.split(key, len(opt_state.params))
        xs_samples = jnp.stack([self.domain[name].sample(key, (size,))
                                for key, name in zip(keys,opt_state.params)], axis=1)

        # Use the acquisition function to find the best parameters
        zs, (means, stds) = self.acq(xs_samples, xs, ys, mask, gpparams)
        idx = jnp.argmax(zs)
        best_params = jax.tree_util.tree_map(lambda d: d[idx], samples)
        if has_prior:
            return best_params, (xs_samples, means, stds)
        return best_params

    def expand(self, opt_state):
        current = jnp.sum(opt_state.mask)

        if current == len(opt_state.mask):
            pad_value = int(np.ceil(len(opt_state.mask)*2 / 10) * 10)
            diff = pad_value - len(opt_state.mask)
            mask = jnp.pad(opt_state.mask, (0, diff))
            ys = jnp.pad(opt_state.ys, (0, diff))
            params = {}
            for key in opt_state.params:
                params[key] = jnp.pad(opt_state.params[key], (0, diff))
        else:
            mask = opt_state.mask
            ys = opt_state.ys
            params = opt_state.params

        opt_state = OptimizerState(params=params, ys=ys, best_score=opt_state.best_score,
                                   best_params=opt_state.best_params, mask=mask,
                                   gp_state=opt_state.gp_state)
        return opt_state


    def fit(self, opt_state, y, new_params):
        """
        Updates the optimizer state with a new observation.

        Parameters
        ----------
        opt_state : OptimizerState
            The current state of the optimizer.
        y : float
            The objective function value for the new observation.
        new_params : dict
            The parameters corresponding to the new observation.

        Returns
        -------
        OptimizerState
            The updated state of the optimizer including the new observation.
        """
        opt_state = self.expand(opt_state) # Prompts recompilation
        opt_state = self._fit(opt_state, y, new_params)
        return opt_state


    @partial(jax.jit, static_argnums=(0,))
    def _fit(self, opt_state, y, new_params):
        last_idx = jnp.arange(len(opt_state.mask)) == jnp.argmin(opt_state.mask)
        mask = jnp.asarray(jnp.where(last_idx, True, opt_state.mask))
        ys = jnp.where(last_idx, y, opt_state.ys)
        params = jax.tree_util.tree_map(lambda x, y: jnp.where(last_idx, y, x), opt_state.params, new_params)

        xs = jnp.stack([self.domain[key].transform(params[key])
                        for key in params], axis=1)
        gp_state = posterior_fit(ys, xs, mask=mask, state=opt_state.gp_state)

        best_score = self.best_fn(ys, where=mask, initial=self.initial)
        best_params_idx = self.best_params_fn(jnp.where(mask, ys, self.initial))
        best_params = jax.tree_util.tree_map(lambda x: x[best_params_idx], params)

        opt_state = OptimizerState(params=params, ys=ys, best_score=best_score,
                                   best_params=best_params, mask=mask,
                                   gp_state=gp_state)
        return opt_state

