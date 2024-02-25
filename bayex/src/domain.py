import jax


class Domain:
    def __init__(self, dtype):
        self.dtype = dtype

    def __hash__(self):
        return hash(self.dtype)

    def __eq__(self, other):
        return self.dtype == other.dtype

    def transform(self, x):
        raise NotImplementedError

    def sample(self, key, shape):
        raise NotImplementedError


class Real(Domain):
    def __init__(self, lower, upper):
        assert (isinstance(lower, float) or isinstance(lower, int)), "Lower bound must be a float"
        assert (isinstance(upper, float) or isinstance(lower, int)), "Upper bound must be a float"
        assert lower < upper, "Lower bound must be less than upper bound"

        self.lower = float(lower)
        self.upper = float(upper)
        super().__init__(dtype=jax.numpy.float32)

    def __hash__(self):
        return hash((self.lower, self.upper))

    def __eq__(self, other):
        return self.lower == other.lower and self.upper == other.upper

    def transform(self, x):
        return jax.numpy.clip(x, self.lower, self.upper)

    def sample(self, key, shape):
        samples = jax.random.uniform(key, shape, minval=self.lower, maxval=self.upper)
        return self.transform(samples)


class Integer(Domain):
    def __init__(self, lower, upper):
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

    def transform(self, x):
        return jax.numpy.clip(jax.numpy.round(x), self.lower, self.upper).astype(jax.numpy.float32)

    def sample(self, key, shape):
        samples = jax.random.randint(key, shape, minval=self.lower, maxval=self.upper + 1)
        return self.transform(samples)

