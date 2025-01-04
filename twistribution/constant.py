from typing import Any
from twistribution.distribution import DiscreteDistribution


class Constant(DiscreteDistribution):
    """
    A constant value with a probability of 1.
    """

    def __init__(self, value: float):
        super().__init__(equality_tolerance=0)
        self.value = value

    def parameters(self) -> tuple[Any, ...]:
        return (self.value,)

    def pmf(self, x: float) -> float:
        return 1 if x == self.value else 0

    def cdf(self, x: float) -> float:
        return 1 if x >= self.value else 0

    @property
    def mean(self) -> float:
        return self.value

    @property
    def variance(self) -> float:
        return 0

    def __lt__(self, other):
        if isinstance(other, Constant):
            return Constant(int(self.value < other.value))
        if isinstance(other, (int, float)):
            return Constant(int(self.value < other))
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Constant):
            return Constant(int(self.value <= other.value))
        if isinstance(other, (int, float)):
            return Constant(int(self.value <= other))
        return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, Constant):
            return NotImplemented
        return self.value == other.value

    def __float__(self) -> float:
        """Implicit conversion to float."""
        return float(self.value)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Constant(self.value + other)
        if isinstance(other, Constant):
            return Constant(self.value + other.value)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Constant(self.value - other)
        if isinstance(other, Constant):
            return Constant(self.value - other.value)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Constant(self.value * other)
        if isinstance(other, Constant):
            return Constant(self.value * other.value)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Constant(self.value / other)
        if isinstance(other, Constant):
            return Constant(self.value / other.value)
        return NotImplemented
