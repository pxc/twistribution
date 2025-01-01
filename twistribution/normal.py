from math import erf, sqrt, pi, exp
from typing import Any

from twistribution.bernoulli import Bernoulli
from twistribution.constants import DEFAULT_EQUALITY_TOLERANCE
from twistribution.distribution import ContinuousDistribution


class Normal(ContinuousDistribution):
    def __init__(
        self,
        mean: float,
        stddev: float,
        equality_tolerance: float = DEFAULT_EQUALITY_TOLERANCE,
    ):
        super().__init__(equality_tolerance)
        if stddev < 0:
            raise ValueError("Standard deviation must be non-negative")
        self.mean = mean
        self.stddev = stddev

    def parameters(self) -> tuple[Any, ...]:
        return self.mean, self.stddev

    def pdf(self, x: float) -> float:
        return normal_pdf(x=x, mean=self.mean, variance=self.stddev**2)

    def cdf(self, x: float) -> float:
        return (1 + erf(x / sqrt(2))) / 2

    def __lt__(self, other):
        if isinstance(other, Normal):
            combined_stddev = sqrt(self.stddev**2 + other.stddev**2)
            z = (self.mean - other.mean) / combined_stddev
            # Use the standard Normal CDF approximation with the Error function erf
            p = 0.5 * (1 + erf(z / sqrt(2)))
            return Bernoulli(1 - p)
        elif isinstance(other, (int, float)):
            z = (self.mean - other) / self.stddev
            # Use the standard Normal CDF approximation with the Error function erf
            p = 0.5 * (1 + erf(z / sqrt(2)))
            return Bernoulli(1 - p)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Normal(self.mean + other, self.stddev)
        elif isinstance(other, Normal):
            return Normal(
                self.mean + other.mean, (self.stddev**2 + other.stddev**2) ** 0.5
            )
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Normal(self.mean - other, self.stddev)
        elif isinstance(other, Normal):
            return Normal(
                self.mean - other.mean, (self.stddev**2 + other.stddev**2) ** 0.5
            )
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Normal(self.mean * other, abs(self.stddev * other))
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return Normal(self.mean / other, self.stddev / abs(other))
        return NotImplemented


def normal_pdf(x, mean, variance):
    coefficient = 1.0 / sqrt(2 * pi * variance)
    exponent = -((x - mean) ** 2) / (2 * variance)
    return coefficient * exp(exponent)
