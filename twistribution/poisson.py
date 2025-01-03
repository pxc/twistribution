import math
from typing import Any

from twistribution.bernoulli import Bernoulli
from twistribution.constants import DEFAULT_EQUALITY_TOLERANCE
from twistribution.discrete import Discrete
from twistribution.distribution import DiscreteDistribution


class Poisson(DiscreteDistribution):
    def __init__(
        self, mean: float, equality_tolerance: float = DEFAULT_EQUALITY_TOLERANCE
    ):
        super().__init__(equality_tolerance)
        if mean <= 0:
            raise ValueError(f"Mean must be strictly positive; got {mean}")
        self.mean = mean

    def parameters(self) -> tuple[Any, ...]:
        return (self.mean,)

    @property
    def variance(self) -> float:
        return self.mean

    def pmf(self, x: float | int) -> float:
        if not float(x).is_integer() or x < 0:
            return 0.0
        k = int(x)
        return self.mean**k * math.exp(-self.mean) / math.factorial(k)

    def cdf(self, x: float | int) -> float:
        if x < 0:
            return 0.0
        if math.isinf(x):
            return 1.0
        k = math.floor(x)
        return sum(
            self.mean**i * math.exp(-self.mean) / math.factorial(i)
            for i in range(k + 1)
        )

    def to_discrete(self, min_probability: float = 1e-10) -> Discrete:
        values = {}
        k = 0
        while True:
            p = self.pmf(k)
            if p < min_probability:
                break
            values[k] = p
            k += 1
        return Discrete(values)

    def __lt__(self, other):
        if isinstance(other, Poisson):
            return self.to_discrete() < other.to_discrete()
        if isinstance(other, (float, int)):
            return Bernoulli(self.cdf(other) - self.pmf(other))
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Poisson):
            return self.to_discrete() <= other.to_discrete()
        if isinstance(other, (float, int)):
            return Bernoulli(self.cdf(other))
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Poisson):
            return Poisson(self.mean + other.mean)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)
