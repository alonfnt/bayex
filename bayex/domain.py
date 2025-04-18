from typing import Tuple
import jax


class Domain:
    def __init__(self, dtype):
        self.dtype = dtype

    def __hash__(self):
        return hash(self.dtype)

    def __eq__(self, other):
        return self.dtype == other.dtype

    def transform(self, x: jax.Array):
        raise NotImplementedError

    def sample(self, key: jax.Array, shape: Tuple):
        raise NotImplementedError


class Real(Domain):
    """
    Continuous real-valued domain with clipping.

    Represents a parameter that can take real values within [lower, upper].
    """

    def __init__(self, lower, upper):
        """
        Initializes a real domain with bounds.

        Args:
            lower: Lower bound (inclusive).
            upper: Upper bound (inclusive).
        """
        assert isinstance(lower, float) or isinstance(lower, int), "Lower bound must be a float"
        assert isinstance(upper, float) or isinstance(lower, int), "Upper bound must be a float"
        assert lower < upper, "Lower bound must be less than upper bound"

        self.lower = float(lower)
        self.upper = float(upper)
        super().__init__(dtype=jax.numpy.float32)

    def __hash__(self):
        return hash((self.lower, self.upper))

    def __eq__(self, other):
        return self.lower == other.lower and self.upper == other.upper

    def transform(self, x: jax.Array):
        """
        Clips values to the domain range [lower, upper].

        Args:
            x: Input values.

        Returns:
            Clipped values within bounds.
        """
        return jax.numpy.clip(x, self.lower, self.upper)

    def sample(self, key: jax.Array, shape: Tuple):
        """
        Samples uniformly from the domain.

        Args:
            key: JAX PRNGKey.
            shape: Desired output shape.

        Returns:
            Sampled values clipped to the domain.
        """
        samples = jax.random.uniform(key, shape, minval=self.lower, maxval=self.upper)
        return self.transform(samples)


class Integer(Domain):
    """
    Discrete integer-valued domain with rounding and clipping.

    Represents a parameter that can take integer values within [lower, upper].
    """

    def __init__(self, lower, upper):
        """
        Initializes an integer domain with bounds.

        Args:
            lower: Lower integer bound (inclusive).
            upper: Upper integer bound (inclusive).
        """
        assert isinstance(lower, int), "Lower bound must be an integer"
        assert isinstance(upper, int), "Upper bound must be an integer"
        assert lower < upper, "Lower bound must be less than upper bound"

        self.lower = int(lower)
        self.upper = int(upper)
        super().__init__(dtype=jax.numpy.int32)

    def __hash__(self):
        return hash((self.lower, self.upper))

    def __eq__(self, other):
        return self.lower == other.lower and self.upper == other.upper

    def transform(self, x: jax.Array):
        """
        Rounds and clips values to the integer domain.

        Args:
            x: Input values.

        Returns:
            Rounded and clipped values as float32.
        """
        return jax.numpy.clip(jax.numpy.round(x), self.lower, self.upper).astype(jax.numpy.float32)

    def sample(self, key: jax.Array, shape: Tuple):
        """
        Samples integers uniformly from the domain.

        Args:
            key: JAX PRNGKey.
            shape: Desired output shape.

        Returns:
            Sampled values clipped to valid integer range.
        """
        samples = jax.random.randint(key, shape, minval=self.lower, maxval=self.upper + 1)
        return self.transform(samples)


class ParamSpace:
    """
    Internal class that manages a collection of named parameter domains.

    This utility encapsulates logic for sampling, transforming, and handling
    structured parameter inputs defined by a mapping of variable names to Domain
    instances (e.g., Real, Integer).

    Example:
        >>> space = ParamSpace({
        ...     "x1": Real(0.0, 1.0),
        ...     "x2": Integer(1, 5)
        ... })
        >>> key = jax.random.PRNGKey(0)
        >>> samples = space.sample_tree(key, (128,))
        >>> xs = space.transform_tree(samples)

    Notes:
        This class is intended for internal use by the optimizer and should not
        be exposed as part of the public API.
    """

    def __init__(self, space: dict):
        self.space = space

    def sample_params(self, key: jax.Array, shape: Tuple) -> dict:
        keys = jax.random.split(key, len(self.space))
        return {name: self.space[name].sample(k, shape) for name, k in zip(self.space, keys)}

    def to_array(self, tree: dict) -> jax.Array:
        """
        Transforms a batch of parameter values into a 2D array suitable for GP input.

        Applies each domain's `.transform()` to its corresponding parameter values.

        Args:
            tree: A dictionary of parameter name → array of raw values.

        Returns:
            A JAX array of shape (batch_size, num_params) with transformed values.
        """
        return jax.numpy.stack([self.space[k].transform(tree[k]) for k in self.space], axis=1)


    def to_dict(self, xs: jax.Array) -> dict:
        """
        Converts a stacked parameter matrix back into named parameter trees.

        Typically used after optimization in transformed space.

        Args:
            xs: A 2D JAX array of shape (batch_size, num_params), with each column
                corresponding to a parameter.

        Returns:
            A dictionary mapping parameter names to individual 1D arrays.
        """
        return {k: self.space[k].transform(xs[:, i]) for i, k in enumerate(self.space)}