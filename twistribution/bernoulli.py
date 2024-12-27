from typing import Any

from twistribution.distribution import DiscreteDistribution


class Bernoulli(DiscreteDistribution):
    def __init__(self, p: float):
        if not (0 <= p <= 1):
            raise ValueError("Probability p must be in the range [0, 1]")
        self.p = p

    def parameters(self) -> tuple[Any, ...]:
        return (self.p,)

    def mean(self):
        return self.p

    def variance(self):
        return self.p * (1 - self.p)
