from typing import Any

from twistribution.constants import DEFAULT_EQUALITY_TOLERANCE
from twistribution.distribution import DiscreteDistribution


class Bernoulli(DiscreteDistribution):
    def __init__(
        self, p: float, equality_tolerance: float = DEFAULT_EQUALITY_TOLERANCE
    ):
        super().__init__(equality_tolerance)

        if p < 0:
            if p >= -self.equality_tolerance:
                p = 0
            else:
                raise ValueError(f"Probability p must be in the range [0, 1]; got {p}.")
        elif p > 1:
            if p <= 1 + self.equality_tolerance:
                p = 1
            else:
                raise ValueError(f"Probability p must be in the range [0, 1]; got {p}.")
        self.p = p

    def parameters(self) -> tuple[Any, ...]:
        return (self.p,)

    def mean(self):
        return self.p

    def variance(self):
        return self.p * (1 - self.p)

    def __lt__(self, other):
        if isinstance(other, (float, int)):
            if other <= 0:
                return Bernoulli(0)
            elif other > 1:
                return Bernoulli(1)
            else:
                return Bernoulli(1 - self.p)
        elif isinstance(other, Bernoulli):
            this_0_and_other_1 = (1 - self.p) * other.p
            return Bernoulli(this_0_and_other_1)
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, (float, int)):
            if other < 0:
                return Bernoulli(0)
            elif other >= 1:
                return Bernoulli(1)
            else:
                return Bernoulli(1 - self.p)
        elif isinstance(other, Bernoulli):
            this_0 = 1 - self.p
            this_1_and_other_1 = self.p * other.p
            return Bernoulli(this_0 + this_1_and_other_1)
        return NotImplemented
