import pytest
import jax
import jax.numpy as jnp
from bayex.domain import Real, Integer


@pytest.mark.parametrize("lower, upper", [(0.0, 1.0), (-5.5, 5.5)])
def test_real_transform_clips_within_bounds(lower, upper):
    domain = Real(lower, upper)
    x = jnp.array([lower - 1, (lower + upper) / 2, upper + 1])
    out = domain.transform(x)
    assert jnp.all(out >= lower) and jnp.all(out <= upper)


def test_real_sample_within_bounds():
    domain = Real(0.0, 1.0)
    key = jax.random.PRNGKey(0)
    shape = (100,)
    samples = domain.sample(key, shape)
    assert samples.shape == shape
    assert samples.dtype == jnp.float32
    assert jnp.all(samples >= 0.0) and jnp.all(samples <= 1.0)


@pytest.mark.parametrize("lower, upper", [(0, 5), (-3, 3)])
def test_integer_transform_and_type(lower, upper):
    domain = Integer(lower, upper)
    x = jnp.array([lower - 2.3, (lower + upper) / 2.0, upper + 2.7])
    out = domain.transform(x)
    assert jnp.issubdtype(out.dtype, jnp.floating)
    assert jnp.all(out >= lower) and jnp.all(out <= upper)
    assert jnp.all(out == jnp.round(out))


def test_integer_sample_within_bounds():
    domain = Integer(1, 10)
    key = jax.random.PRNGKey(42)
    shape = (1000,)
    samples = domain.sample(key, shape)
    assert samples.shape == shape
    assert jnp.all(samples >= 1) and jnp.all(samples <= 10)
    assert jnp.all(jnp.equal(samples, jnp.round(samples)))


def test_domain_equality_and_hash():
    a = Real(0.0, 1.0)
    b = Real(0.0, 1.0)
    c = Real(1.0, 2.0)
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    assert hash(a) != hash(c)

    i1 = Integer(1, 5)
    i2 = Integer(1, 5)
    i3 = Integer(0, 4)
    assert i1 == i2
    assert i1 != i3
    assert hash(i1) == hash(i2)
    assert hash(i1) != hash(i3)


@pytest.mark.parametrize("lower, upper", [
    (1.0, 1.0),         # equal
    ("a", 1.0),         # wrong type
    (1.0, "b"),         # wrong type
    (5.0, 3.0),         # lower > upper
])
def test_real_init_invalid(lower, upper):
    with pytest.raises(AssertionError):
        Real(lower, upper)


@pytest.mark.parametrize("lower, upper", [
    (5, 5),             # equal
    ("a", 5),           # wrong type
    (0, "b"),           # wrong type
    (10, 1),            # lower > upper
])
def test_integer_init_invalid(lower, upper):
    with pytest.raises(AssertionError):
        Integer(lower, upper)
