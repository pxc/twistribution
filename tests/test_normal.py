import math

from twistribution.bernoulli import Bernoulli
from twistribution.normal import Normal


def test_adding_normal_and_scalar():
    a = Normal(0, 1)
    b = 2
    added = a + b
    assert added == Normal(2, 1)


def test_adding_scalar_and_normal():
    a = 2
    b = Normal(0, 1)
    added = a + b
    assert added == Normal(2, 1)


def test_adding_two_normals():
    a = Normal(0, 1)
    b = Normal(2, 3)
    added = a + b
    assert added == Normal(2, math.sqrt(10))


def test_subtracting_two_normals():
    a = Normal(5, 2)
    b = Normal(3, 1)
    subtracted = a - b
    assert subtracted == Normal(2, math.sqrt(5))


def test_multiplying_normal_by_scalar():
    a = Normal(1, 2)
    b = 3
    multiplied = a * b
    assert multiplied == Normal(3, 6)


def test_multiplying_scalar_by_normal():
    a = 3
    b = Normal(1, 2)
    multiplied = a * b
    assert multiplied == Normal(3, 6)


def test_dividing_normal_by_scalar():
    a = Normal(4, 8)
    b = 2
    divided = a / b
    assert divided == Normal(2, 4)


def test_normal_ge_bernoulli1():
    a = Normal(0, 1)
    b = Normal(0, 1)
    result = a >= b
    assert result == Bernoulli(0.5)


def test_normal_ge_bernoulli2():
    a = Normal(-10, 1)
    b = Normal(0, 1)
    result = a >= b
    assert result.p < 1e-6


def test_normal_ge_bernoulli3():
    a = Normal(10, 1)
    b = Normal(0, 1)
    result = a >= b
    assert result.p > 0.99999


def test_normal_gt_bernoulli4():
    # based on the concrete example here:
    # https://math.stackexchange.com/questions/235012/probability-of-one-normdist-being-greater-than-another
    a = Normal(657, 3)
    b = Normal(661, 2)
    result = a > b
    assert 0.13362 < result.p < 0.13364


def test_normal_gt_scalar1():
    a = Normal(1, 2)
    result = a > 0
    assert isinstance(result, Bernoulli)
    assert 0.691 < result.p < 0.692


def test_normal_gt_scalar2():
    a = Normal(0, 1)
    result = a > 0
    assert isinstance(result, Bernoulli)
    assert result == Bernoulli(0.5)
