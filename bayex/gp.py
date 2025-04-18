from collections import namedtuple
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky, solve_triangular
import optax

MASK_VARIANCE = 1e12 # High variance for masked points to not affect the process.

GPParams = namedtuple("GPParams", ["noise", "amplitude", "lengthscale"])
GPState = namedtuple("GPState", ["params", "momentums", "scales"])

def exp_quadratic(x1, x2, mask):
    distance = jnp.sum((x1 - x2) ** 2)
    return jnp.exp(-distance) * mask


def cov(x1, x2, mask1, mask2):
    M = jnp.outer(mask1, mask2)
    k = exp_quadratic
    return jax.vmap(jax.vmap(k, in_axes=(None, 0, 0)), in_axes=(0, None, 0))(x1, x2, M)


def softplus(x):
    return jnp.logaddexp(x, 0.0)


def gaussian_process(
    params,
    x: jnp.ndarray,
    y: jnp.ndarray,
    mask,
    xt: Optional[jnp.ndarray] = None,
    compute_ml: bool = False,
) -> Any:
    # Number of points in the prior distribution
    n = x.shape[0]

    noise, amp, ls = jax.tree_util.tree_map(softplus, params)

    ymean = jnp.mean(y, where=mask.astype(bool))
    y = (y - ymean) * mask
    x = x / ls
    K = amp * cov(x, x, mask, mask) + (jnp.eye(n) * (noise + 1e-6))
    K += jnp.eye(n) * (1.0 - mask.astype(float)) * MASK_VARIANCE
    L = cholesky(K, lower=True)
    K_inv_y = solve_triangular(L.T, solve_triangular(L, y, lower=True), lower=False)

    if compute_ml:
        logp = 0.5 * jnp.dot(y.T, K_inv_y)
        logp += jnp.sum(jnp.log(jnp.diag(L)))
        logp -= jnp.sum(1.0 - mask) * 0.5 * jnp.log(MASK_VARIANCE)
        logp += (jnp.sum(mask) / 2) * jnp.log(2 * jnp.pi)
        logp += jnp.sum(-0.5 * jnp.log(2*jnp.pi) - jnp.log(amp) - jnp.log(amp)**2)
        return jnp.sum(logp)

    assert xt is not None, "xt can't be None during prediction."
    xt = xt / ls

    # Compute the covariance with the new point xt
    mask_t = jnp.ones(len(xt))==1
    K_cross = amp * cov(x, xt, mask, mask_t)

    K_inv_y = K_inv_y * mask # masking
    pred_mean = jnp.dot(K_cross.T, K_inv_y) + ymean
    v = solve_triangular(L, K_cross, lower=True)
    pred_var = amp * cov(xt, xt, mask_t, mask_t) - v.T @ v
    pred_std = jnp.sqrt(jnp.maximum(jnp.diag(pred_var), 1e-10))
    return pred_mean, pred_std


marginal_likelihood = partial(gaussian_process, compute_ml=True)
grad_fun = jax.jit(jax.grad(marginal_likelihood))
predict = jax.jit(partial(gaussian_process, compute_ml=False))

def neg_log_likelihood(params, x, y, mask):
    ll = marginal_likelihood(params, x, y, mask)

    # Weak priors to keep things sane
    # params = jax.tree.map(softplus, params)
    priors = GPParams(-8.0, 1.0, 1.0)
    log_prior = jax.tree.map(lambda p, m: jnp.sum((p - m) ** 2), params, priors)
    log_prior = sum(jax.tree.leaves(log_prior))
    log_posterior = ll - 0.5 * log_prior
    return -log_posterior


def posterior_fit(
    y: jax.Array,
    x: jax.Array,
    mask: jax.Array,
    params: GPParams,
    lr: float = 1e-3,
    trainsteps: int = 100,
) -> GPState:

    optimizer = optax.chain(optax.clip_by_global_norm(10.0), optax.adamw(lr))
    opt_state = optimizer.init(params)

    def train_step(carry, _):
        params, opt_state = carry
        grads = jax.grad(neg_log_likelihood)(params, x, y, mask)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), None

    (params, _), __ = jax.lax.scan(train_step, (params, opt_state), None, length=trainsteps)
    return params
