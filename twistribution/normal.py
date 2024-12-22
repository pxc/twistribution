import math

from twistribution.bernoulli import Bernoulli
from twistribution.distribution import ContinuousDistribution


class Normal(ContinuousDistribution):
    def __init__(self, mean: float, stddev: float):
        if stddev < 0:
            raise ValueError("Standard deviation must be non-negative")
        self.mean = mean
        self.stddev = stddev

    def __str__(self):
        return f"Normal({self.mean}, {self.stddev})"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, Normal):
            return NotImplemented
        return self.mean == other.mean and self.stddev == other.stddev

    def __add__(self, other):
        if isinstance(other, Normal):
            return Normal(
                self.mean + other.mean, (self.stddev**2 + other.stddev**2) ** 0.5
            )
        else:
            return Normal(self.mean + other, self.stddev)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Normal):
            return Normal(
                self.mean - other.mean, (self.stddev**2 + other.stddev**2) ** 0.5
            )
        else:
            return Normal(self.mean - other, self.stddev)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Normal(self.mean * other, abs(self.stddev * other))
        elif isinstance(other, Normal):
            raise NotImplementedError(
                "Multiplication of two Normals is not defined in a simple way."
            )
        else:
            raise TypeError("Can only multiply a Normal distribution by a scalar")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return Normal(self.mean / other, self.stddev / abs(other))
        else:
            raise TypeError("Can only divide a Normal distribution by a scalar")

    def __gt__(self, other):
        if isinstance(other, Normal):
            from math import sqrt, erf

            combined_stddev = sqrt(self.stddev**2 + other.stddev**2)
            z = (self.mean - other.mean) / combined_stddev
            # Use the standard Normal CDF approximation with the Error function erf
            p = 0.5 * (1 + erf(z / sqrt(2)))
            return Bernoulli(p)
        elif isinstance(other, (int, float)):
            from math import sqrt, erf

            z = (self.mean - other) / self.stddev
            # Use the standard Normal CDF approximation with the Error function erf
            p = 0.5 * (1 + erf(z / sqrt(2)))
            return Bernoulli(p)
        else:
            raise TypeError(
                "Can only compare a Normal distribution with another Normal or a scalar"
            )


def normal_pdf(x, mean, variance):
    coefficient = 1.0 / math.sqrt(2 * math.pi * variance)
    exponent = -((x - mean) ** 2) / (2 * variance)
    return coefficient * math.exp(exponent)
