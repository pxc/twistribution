import pytest
import random
import math

from tests.utils_for_testing import close, estimate_probability
from twistribution.bernoulli import Bernoulli
from twistribution.poisson import Poisson


def test_poisson_creation():
    # Test valid creation
    p = Poisson(1.5)
    assert p.mean == 1.5

    # Test invalid parameters
    with pytest.raises(ValueError):
        Poisson(0)
    with pytest.raises(ValueError):
        Poisson(-1)


def test_poisson_equality():
    # Test exact equality
    assert Poisson(2.0) == Poisson(2.0)
    assert Poisson(2.0) != Poisson(2.1)

    # Test approximate equality
    p1 = Poisson(2.0)
    p2 = Poisson(2.0 + 1e-10)
    assert p1 == p2

    # Test inequality with different types
    assert Poisson(2.0) != "not a distribution"


def test_poisson_parameters():
    p = Poisson(3.5)
    assert p.parameters() == (3.5,)


def test_poisson_str_repr():
    p = Poisson(2.5)
    assert str(p) == "Poisson(2.5)"
    assert repr(p) == "Poisson(2.5)"


def test_poisson_mean_variance():
    p = Poisson(4.2)
    assert p.mean == 4.2
    assert p.variance == 4.2


def test_poisson_comparison():
    p1 = Poisson(2.0)
    p2 = Poisson(3.0)

    # Test comparisons with other Poissons
    assert p1 < p2
    assert p1 <= p2
    assert p2 > p1
    assert p2 >= p1

    # Test comparisons with numbers
    assert p1 > 0
    assert p1 >= 0
    assert p2 < float("inf")
    assert p2 <= float("inf")

    # Test invalid comparisons
    with pytest.raises(TypeError):
        _ = p1 < "not a number"


def test_poisson_comparison_with_self():
    p = Poisson(2.0)

    expected_le_prob = compare_poissons_le(p, p)
    result_le_prob = (p <= p).p
    assert close(
        expected_le_prob, result_le_prob, tolerance=1e-3
    ), f"Expected {expected_le_prob}, got {result_le_prob}"

    expected_lt_prob = compare_poissons_lt(p, p)
    result_lt_prob = (p < p).p
    assert close(
        expected_lt_prob, result_lt_prob, tolerance=1e-3
    ), f"Expected {expected_lt_prob}, got {result_lt_prob}"


def generate_poisson(mean):
    """Generate a Poisson random variable using the inverse transform method"""
    L = math.exp(-mean)
    k = 0
    p = 1.0

    while True:
        p *= random.random()
        if p < L:
            return k
        k += 1


@estimate_probability()
def compare_poisson_to_scalar(p, scalar):
    return generate_poisson(p.mean) < scalar


@estimate_probability()
def compare_poissons_lt(p1, p2):
    return generate_poisson(p1.mean) < generate_poisson(p2.mean)


@estimate_probability()
def compare_poissons_le(p1, p2):
    return generate_poisson(p1.mean) <= generate_poisson(p2.mean)


def test_poisson_comparison_by_sampling():
    # Test comparison with scalar
    p = Poisson(2.0)
    for x in [1, 2, 3]:
        sampled = compare_poisson_to_scalar(p, x)
        calculated = p < x
        assert isinstance(calculated, Bernoulli)
        assert close(
            calculated.p, sampled, tolerance=1e-3
        ), f"Expected {sampled}, got {calculated.p}"

    # Test comparison between Poissons
    p1 = Poisson(2.0)
    p2 = Poisson(3.0)
    sampled = compare_poissons_lt(p1, p2)
    calculated = p1 < p2
    assert isinstance(calculated, Bernoulli)
    assert close(
        calculated.p, sampled, tolerance=1e-3
    ), f"Expected {sampled}, got {calculated.p}"

    # Test equal means
    p3 = Poisson(2.0)
    sampled = compare_poissons_lt(p1, p3)
    calculated = p1 < p3
    assert isinstance(calculated, Bernoulli)
    assert close(
        calculated.p, sampled, tolerance=1e-3
    ), f"Expected {sampled}, got {calculated.p}"
