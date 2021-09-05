from jax import grad, jit, vmap
import jax.numpy as jnp
import jax.scipy as scipy

from functools import partial

def cov_map(cov_func, xs, xs2=None):
    if xs2 is None:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs)
    else:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs2).T


def softplus(x):
    return jnp.logaddexp(x, 0.0)


def exp_quadratic(x1, x2):
    return jnp.exp(-jnp.sum((x1 - x2) ** 2))


def marginal_likelihood(y, kinvy, chol, N, A):
    log2pi = jnp.log(2.0 * jnp.pi)
    ml = jnp.sum(
        -0.5 * jnp.dot(y.T, kinvy)
        - jnp.sum(jnp.log(jnp.diag(chol)))
        - (N / 2.0) * log2pi
    )
    ml -= jnp.sum(-0.5 * jnp.log(2 * 3.1415) - jnp.log(A) ** 2)
    return ml
  
def gp(A, x, y, xtest=None, train=False):
    # Some X points are given.
    numpts = x.shape[0]
    eye = jnp.eye(numpts)
    
    # A is the only parameter to 'train'
    A = softplus(A)

    ymean = jnp.mean(y)
    y = y - ymean

    train_cov = A * cov_map(exp_quadratic, x) + eye * (1e-6)
    chol = scipy.linalg.cholesky(train_cov, lower=True)
    kinvy = scipy.linalg.solve_triangular(
        chol.T, scipy.linalg.solve_triangular(chol, y, lower=True)
    )

    if train:
        return marginal_likelihood(y, kinvy, chol, numpts, A)

    cross_cov = A * cov_map(exp_quadratic, x, xtest)
    mu = jnp.dot(cross_cov.T, kinvy) + ymean
    v = scipy.linalg.solve_triangular(chol, cross_cov, lower=True)
    var = A * cov_map(exp_quadratic, xtest) - jnp.dot(v.T, v)
    if numpts > 1:
        var = jnp.diag(var)
    std = jnp.sqrt(var)
    return mu, std


grad_fun = jit(grad(partial(gp, train=True)))
predict = jit(partial(gp, train=False))


def train(A, p, scale, x, y, lr=0.01, nsteps=5):
    grads = grad_fun(A, x, y)
    for _ in range(nsteps):
        p = 0.9 * p + 0.1 * grads[0]
        scale = 0.9 * scale + 0.1 * grads[0] ** 2
        A -= lr * p / jnp.sqrt(scale + 1e-5)
    return A, p, scale
