from twistribution.bernoulli import Bernoulli


def test_bernoulli_mean_variance():
    b = Bernoulli(0.5)
    assert b.mean() == 0.5
    assert b.variance() == 0.25
