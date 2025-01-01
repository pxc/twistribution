import pytest

from tests.utils_for_testing import close
from twistribution.bernoulli import Bernoulli
from twistribution.discrete import Discrete
from twistribution.normal import normal_pdf


def test_keys_not_in_order():
    with pytest.raises(ValueError):
        _ = Discrete({1: 0.5, 0: 0.5})


def test_bad_probabilities():
    with pytest.raises(ValueError):
        _ = Discrete({1: 0.5, 2: 0.51})


def test_add_scalar():
    discrete = Discrete({1: 0.25, 2: 0.25, 4: 0.25, 5: 0.25})
    updated = discrete + 1.5
    assert updated == Discrete({2.5: 0.25, 3.5: 0.25, 5.5: 0.25, 6.5: 0.25})


class TestFairCoinToss:
    """
    Tests of a fair coin, represented as a Discrete() where 0 represents Tails and 1 represents Heads.
    """

    def setup_method(self):
        self.coin = Discrete({0: 0.5, 1: 0.5})

    def test_median(self):
        assert self.coin.median() == 0.5

    def test_mean(self):
        assert self.coin.mean() == 0.5

    def test_sum_of_two_coins(self):
        two_coins = self.coin + self.coin
        assert isinstance(two_coins, Discrete)
        assert two_coins == Discrete({0: 0.25, 1: 0.5, 2: 0.25})

    def test_sum_of_three_coins(self):
        three_coins = self.coin + self.coin + self.coin
        assert isinstance(three_coins, Discrete)
        assert three_coins == Discrete({0: 0.125, 1: 0.375, 2: 0.375, 3: 0.125})

    def test_sum_of_three_coins_probabilities(self):
        three_coins = self.coin + self.coin + self.coin
        assert isinstance(three_coins, Discrete)

        prob_greater_than_1 = three_coins > 1
        assert prob_greater_than_1 == Bernoulli(0.5), str(prob_greater_than_1)

        prob_greater_than_or_equal_1 = three_coins >= 1
        assert prob_greater_than_or_equal_1 == Bernoulli(0.875), str(
            prob_greater_than_or_equal_1
        )


class TestFairDice:
    def setup_method(self):
        self.d6 = Discrete({1: 1 / 6, 2: 1 / 6, 3: 1 / 6, 4: 1 / 6, 5: 1 / 6, 6: 1 / 6})

    def test_median(self):
        assert self.d6.median() == 3.5

    def test_mean(self):
        assert self.d6.mean() == (1 + 2 + 3 + 4 + 5 + 6) / 6

    def test_sum_of_two_dice(self):
        two_dice = self.d6 + self.d6
        assert isinstance(two_dice, Discrete)
        assert two_dice == Discrete(
            {
                2: 1 / 36,
                3: 2 / 36,
                4: 3 / 36,
                5: 4 / 36,
                6: 5 / 36,
                7: 6 / 36,
                8: 5 / 36,
                9: 4 / 36,
                10: 3 / 36,
                11: 2 / 36,
                12: 1 / 36,
            }
        )

    def test_sum_of_100_dice(self):
        one_hundred_dice = self.d6
        for _ in range(99):
            one_hundred_dice += self.d6
        assert isinstance(one_hundred_dice, Discrete)
        assert list(one_hundred_dice.probabilities.keys()) == list(range(100, 601))
        assert one_hundred_dice.median() == 350
        assert close(one_hundred_dice.mean(), 350)
        assert close(one_hundred_dice.probabilities[100], (1 / 6) ** 100)
        assert close(one_hundred_dice.probabilities[600], (1 / 6) ** 100)

        # sum probability should be approximately Normal(350, 3500/12)
        # see derivation in https://math.stackexchange.com/a/406203/153290
        normal_mean = 350
        normal_variance = 3500 / 12
        for k in range(100, 601):
            normal = normal_pdf(x=k, mean=normal_mean, variance=normal_variance)
            assert close(one_hundred_dice.probabilities[k], normal, tolerance=1e-4)
        # check the tolerance used above isn't too weak (i.e. the max value > 1e-4)
        assert close(max(one_hundred_dice.probabilities.values()), 0.02332260)

    def test_sum_of_two_dice_probabilities(self):
        two_dice = self.d6 + self.d6
        assert isinstance(two_dice, Discrete)

        prob = two_dice >= 7
        assert isinstance(prob, Bernoulli)
        assert close(prob.p, 21 / 36)
        assert (two_dice >= 7) == (two_dice > 6)

        assert (two_dice > 1) == Bernoulli(1.0)
        assert (two_dice > 12) == Bernoulli(0.0)
