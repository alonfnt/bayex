from functools import partial
from typing import Union, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import bayex.acq as boacq
from bayex.gp import GPParams, GPState, posterior_fit


class OptimizerState(NamedTuple):
    params: dict
    ys: Union[jax.Array, np.ndarray]
    best_score: float
    best_params: dict
    mask: jax.Array
    gp_state: GPState


class Optimizer:
    """
    Bayesian optimizer using Gaussian Processes and acquisition functions.

    This class manages the optimization loop for expensive black-box functions
    by modeling them with a Gaussian Process and selecting samples via
    acquisition functions such as EI, PI, UCB, or LCB.
    """

    def __init__(self, domain, acq='EI', maximize=False):
        """
        Initializes the optimizer.

        Args:
            domain: A dict mapping parameter names to domain objects (e.g., Real, Integer).
            acq: Acquisition function ('EI', 'PI', 'UCB', 'LCB').
            maximize: Whether to maximize or minimize the objective.
        """
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
        Initializes the optimizer state from initial data.

        Args:
            ys: Objective values for the initial parameters.
            params: Dict of parameter arrays (same keys as domain).

        Returns:
            Initialized OptimizerState.
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
        Samples new parameters using the acquisition function.

        Args:
            key: JAX PseudoRandom key for random sampling.
            opt_state: Current optimizer state.
            size: Number of samples to draw.
            has_prior: If True, also return GP predictions.

        Returns:
            Sampled parameters (dict), and optionally (xs_samples, means, stds).
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

    def expand(self, opt_state: OptimizerState):
        """
        Expands internal buffers if no space is available.

        Args:
            opt_state: Current optimizer state.

        Returns:
            OptimizerState with expanded storage.
        """
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
        Updates optimizer state with a new observation.

        Args:
            opt_state: Current optimizer state.
            y: New objective value.
            new_params: Parameters that produced y.

        Returns:
            Updated OptimizerState.
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

