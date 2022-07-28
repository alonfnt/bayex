from typing import NamedTuple, Optional

from jax import tree_util
import jax.numpy as jnp


class Measures(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray


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


def update_data(measures: Measures, new_measure: Measures) -> Measures:
    print(new_measure.x.reshape(1, -1).shape)
#    new_x = jnp.stack((measures.x, new_measure.x.reshape(1, -1)), axis=0)
    new_x = measures.x
    new_y = jnp.stack((measures.y, new_measure.y), axis=-1)
    return Measures(x=new_x, y=new_y)
#    return tree_util.tree_map(lambda x, y: jnp.append(x, y.reshape(1, -1), axis=0), measures, new_measure)


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
