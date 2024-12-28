import random
from twistribution.bernoulli import Bernoulli
from twistribution.uniform import Uniform


def estimate_probability(samples=1000000):
    """Decorator that estimates probability by random sampling"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            count = 0
            for _ in range(samples):
                if func(*args, **kwargs):
                    count += 1
            return count / samples

        return wrapper

    return decorator


@estimate_probability()
def compare_uniform_to_scalar(u, scalar):
    return random.uniform(u.a, u.b) < scalar


@estimate_probability()
def compare_uniforms(u1, u2):
    return random.uniform(u1.a, u1.b) < random.uniform(u2.a, u2.b)


def test_adding_uniform_and_scalar():
    a = Uniform(0, 1)
    b = 2
    added = a + b
    assert added == Uniform(2, 3)


def test_multiplying_uniform_and_scalar():
    a = Uniform(1, 2)
    b = 2
    added = a * b
    assert added == Uniform(2, 4)


def assert_close_probability(a, b, tolerance=0.01):
    """Assert that two Bernoulli probabilities are close"""
    assert abs(a.p - b.p) < tolerance, f"Expected {b.p}, got {a.p}"


def test_comparison_against_scalar():
    a = Uniform(0, 1)
    random.seed(42)

    assert_close_probability(a < 0, Bernoulli(compare_uniform_to_scalar(a, 0)))
    assert_close_probability(a < 0.5, Bernoulli(compare_uniform_to_scalar(a, 0.5)))
    assert_close_probability(a < 0.75, Bernoulli(compare_uniform_to_scalar(a, 0.75)))
    assert_close_probability(a < 1, Bernoulli(compare_uniform_to_scalar(a, 1)))


def test_comparison_against_uniform_without_overlap():
    random.seed(42)
    a = Uniform(0, 1)
    b = Uniform(2, 3)

    # When ranges don't overlap, we can be exact
    assert (a < b) == Bernoulli(1.0)
    assert (b < a) == Bernoulli(0.0)


def test_comparison_against_uniform_with_overlap1():
    random.seed(42)
    a = Uniform(0, 1)
    b = Uniform(0, 1)

    prob = compare_uniforms(a, b)
    assert_close_probability(a < b, Bernoulli(prob))


def test_comparison_against_uniform_with_overlap2():
    random.seed(42)
    a = Uniform(0, 1)
    b = Uniform(0.5, 1)

    prob = compare_uniforms(a, b)
    assert_close_probability(a < b, Bernoulli(prob))


def test_comparison_against_uniform_with_overlap3():
    random.seed(42)
    a = Uniform(1, 2)
    b = Uniform(0, 3)

    prob = compare_uniforms(a, b)
    assert_close_probability(a < b, Bernoulli(prob))


def test_comparison_against_uniform_random_pairs():
    # Test 10 random pairs of Uniform distributions
    for _ in range(10):
        # Generate random bounds for two Uniform distributions
        a1, b1 = sorted([random.random(), random.random()])
        a2, b2 = sorted([random.random(), random.random()])

        u1 = Uniform(a1, b1)
        u2 = Uniform(a2, b2)

        # Compare the analytical result with the sampling result
        prob = compare_uniforms(u1, u2)
        assert_close_probability(u1 < u2, Bernoulli(prob))
