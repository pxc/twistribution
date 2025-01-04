from twistribution.bernoulli import Bernoulli
from twistribution.constant import Constant


def test_bernoulli_mean_variance():
    b = Bernoulli(0.5)
    assert b.mean() == 0.5
    assert b.variance() == 0.25


def test_less_than_scalar():
    # Test P(X < 0) = 0 for Bernoulli(p)
    assert (Bernoulli(0.7) < 0) == Constant(0)
    assert (Bernoulli(0.3) < 0) == Constant(0)

    # Test P(X < 0.5) = 0 for Bernoulli(p)
    assert (Bernoulli(0.7) < 0.5) == Bernoulli(0.3), str(Bernoulli(0.3) < 0.5)
    assert (Bernoulli(0.3) < 0.5) == Bernoulli(0.7), str(Bernoulli(0.3) < 0.5)

    # Test P(X < 1) = p for Bernoulli(p)
    assert (Bernoulli(0.7) < 1) == Bernoulli(0.3), str(Bernoulli(0.7) < 1)
    assert (Bernoulli(0.3) < 1) == Bernoulli(0.7), str(Bernoulli(0.3) < 1)

    # Test P(X < 2) = 1 for Bernoulli(p)
    assert (Bernoulli(0.7) < 2) == Constant(1), str(Bernoulli(0.7) < 2)
    assert (Bernoulli(0.3) < 2) == Constant(1), str(Bernoulli(0.3) < 2)


def test_less_than_or_equal_scalar():
    # Test P(X <= 0) = 1-p for Bernoulli(p)
    assert (Bernoulli(0.7) <= 0) == Bernoulli(0.3), str(Bernoulli(0.7) <= 0)
    assert (Bernoulli(0.3) <= 0) == Bernoulli(0.7), str(Bernoulli(0.3) <= 0)

    # Test P(X <= 0.5) = 1-p for Bernoulli(p)
    assert (Bernoulli(0.7) <= 0.5) == Bernoulli(0.3), str(Bernoulli(0.7) <= 0.5)
    assert (Bernoulli(0.3) <= 0.5) == Bernoulli(0.7), str(Bernoulli(0.3) <= 0.5)

    # Test P(X <= 1) = 1 for Bernoulli(p)
    assert (Bernoulli(0.7) <= 1) == Bernoulli(1), str(Bernoulli(0.7) <= 1)
    assert (Bernoulli(0.3) <= 1) == Bernoulli(1), str(Bernoulli(0.3) <= 1)

    # Test P(X <= 2) = 1 for Bernoulli(p)
    assert (Bernoulli(0.7) <= 2) == Bernoulli(1), str(Bernoulli(0.7) <= 2)
    assert (Bernoulli(0.3) <= 2) == Bernoulli(1), str(Bernoulli(0.3) <= 2)
