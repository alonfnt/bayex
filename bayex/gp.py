from collections import namedtuple
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky, solve_triangular

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

    ymean = jnp.mean(y, where=mask)
    y = (y - ymean) * mask
    x = x / ls
    K = amp * cov(x, x, mask, mask) + (jnp.eye(n) * (noise + 1e-6))
    L = cholesky(K, lower=True)
    K_inv_y = solve_triangular(L.T, solve_triangular(L, y, lower=True), lower=False)

    if compute_ml:
        logp = 0.5 * jnp.dot(y.T, K_inv_y)
        logp += jnp.sum(jnp.log(jnp.diag(L)))
        logp += (jnp.sum(mask) / 2) * jnp.log(2 * jnp.pi)
        logp += jnp.sum(-0.5 * jnp.log(2*jnp.pi) - jnp.log(amp) - jnp.log(amp)**2)
        return jnp.sum(logp)

    assert xt is not None, "xt can't be None during prediction."
    xt = xt / ls

    # Compute the covariance with the new point xt
    mask_t = jnp.ones(len(xt))==1
    K_cross = amp * cov(x, xt, mask, mask_t)

    pred_mean = jnp.dot(K_cross.T, K_inv_y) + ymean
    v = solve_triangular(L, K_cross, lower=True)
    pred_var = amp * cov(xt, xt, mask_t, mask_t) - v.T @ v
    pred_std = jnp.sqrt(jnp.diag(pred_var)*(jnp.diag(pred_var)>0))
    return pred_mean, pred_std


marginal_likelihood = partial(gaussian_process, compute_ml=True)
grad_fun = jax.jit(jax.grad(marginal_likelihood))
predict = jax.jit(partial(gaussian_process, compute_ml=False))


def posterior_fit(
    y: jax.Array,
    x: jax.Array,
    mask: jax.Array,
    state: GPState,
    lr: float = 1e-3,
    trainsteps: int = 300,
) -> GPState:

    @jax.jit
    def train_step(i, state):
        params, momentums, scales = state
        grads = grad_fun(params, x, y, mask)

        momentums = jax.tree_util.tree_map(lambda m, g: 0.9 * m + 0.1 * g, momentums, grads)
        scales = jax.tree_util.tree_map(lambda s, g: 0.9 * s + 0.1 * g**2, scales, grads)
        params = jax.tree_util.tree_map(
            lambda p, m, s: p - lr * m / jnp.sqrt(s + 1e-5),
            params,
            momentums,
            scales,
        )
        new_state = GPState(params, momentums, scales)
        return new_state

    state = jax.lax.fori_loop(0, trainsteps, train_step, state)
    return state
