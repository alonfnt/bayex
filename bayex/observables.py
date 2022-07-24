from typing import NamedTuple, Optional

import jax.numpy as jnp
from jax.tree_util import tree_map


class Observable(NamedTuple):
    inputs: jnp.ndarray
    output: jnp.ndarray


class MaskedObservables(NamedTuple):
    inputs: jnp.ndarray
    outputs: jnp.ndarray
    num: int


class DataTypes(NamedTuple):
    integers: list


def extend_array(arr: jnp.ndarray, pad_width: int, axis: int) -> jnp.ndarray:
    """
    Extends the array pad_width only on one direction and fills it with
    the last value of that axis.
    """
    pad_shape = [(0, 0)] * arr.ndim
    pad_shape[axis] = (0, pad_width)  # type: ignore
    return jnp.pad(arr, pad_shape, mode="edge")


def add_observable(observables, new_observable) -> MaskedObservables:
    n = observables.num
    current_obs = Observable(observables.inputs, observables.outputs)
    new_obs = tree_map(lambda x, y: x.at[n].set(y), current_obs, new_observable)
    return MaskedObservables(new_obs.inputs, new_obs.output, num=n + 1)


def round_integers(arr: jnp.ndarray, dtypes: Optional[DataTypes] = None) -> jnp.ndarray:
    """
    The input variables corresponding to an integer-valued input variable are
    rounded to the closest integer value.
    """
    if dtypes is None:
        return arr

    indexes = dtypes.integers
    if indexes:
        integers = jnp.zeros(shape=arr.shape[1])
        integers = integers.at[tuple(indexes)].set(1)
        arr = jnp.where(integers[jnp.newaxis, ...], jnp.round(arr), arr)  # type: ignore
    return arr
