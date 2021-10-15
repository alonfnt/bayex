from enum import Enum
from typing import Any, Callable

import jax.numpy as jnp
from jax.tree_util import tree_map

from bayex.optim import OptimizerParameters
from bayex.types import Array


class Log(Enum):
    ALL = 1
    BEST = 2


def show_sampling(
    x: Array, y: Array, f: Callable, verbose: Log, min_width: int
) -> None:
    """
    Prints the sampled points during the optimization as well as the target
    value.

    Parameters:
    -----------
    x: The sampled points.
    y: The sampled targets.
    f: Target function.
    verbose: Level of detail in the shown information.
             - Log.ALL shows all the sampling points.
             - Log.BEST shows only the iterations that find better results.
    min_width: Minimum width of the column.
    """
    assert x.ndim == 2, "sampled points should be a 2d-array (throws, vars)"
    var_count = x.shape[1]
    assert var_count == f.__code__.co_argcount
    var_names = f.__code__.co_varnames[:var_count]
    max_width = max(tree_map(len, var_names))
    max_width = max_width if max_width > min_width else min_width
    col_template = "{:^" + str(max_width) + "}"
    row_template = col_template * (var_count + 2)
    print(row_template.format("#sample", "target", *var_names))
    col_template = "{:^" + str(max_width) + "g}"
    row_template = col_template * (var_count + 2)

    # I doubt whether is should be better to print in sorted mode
    # or is it better to show in the order they have been sampled
    y_prev = -jnp.inf
    for i, (xi, yi) in enumerate(zip(x, y)):
        if verbose == Log.BEST and yi < y_prev:
            continue
        print(row_template.format(i + 1, yi, *xi))
        y_prev = yi


def show_results(
    res: OptimizerParameters, verbose: Log = Log.ALL, min_width: int = 10
) -> None:
    """
    Prints the sampled points during the optimization as well as the target
    value.

    - This function is a wrapper for `show_samplings`.
    Parameters:
    -----------
    res: The object returned from the optimizer with the optimization
         parameters.
    verbose: Level of detail in the shown information.
             - Log.ALL shows all the sampling points.
             - Log.BEST shows only the iterations that find better results.
    min_width: Minimum width of the column.
    """
    show_sampling(
        x=res.params_all,
        y=res.target_all,
        f=res.f,
        verbose=verbose,
        min_width=min_width,
    )
