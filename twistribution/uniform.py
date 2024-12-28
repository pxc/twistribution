from typing import Any

from twistribution.bernoulli import Bernoulli
from twistribution.distribution import ContinuousDistribution


class Uniform(ContinuousDistribution):
    def __init__(self, a: float, b: float):
        if a >= b:
            raise ValueError("a must be less than b")

        self.a = a
        self.b = b

    def parameters(self) -> tuple[Any, ...]:
        return self.a, self.b

    def pdf(self, x: float) -> float:
        if x < self.a or x > self.b:
            return 0
        return 1 / (self.b - self.a)

    def cdf(self, x: float) -> float:
        if x < self.a:
            return 0
        if x > self.b:
            return 1
        return (x - self.a) / (self.b - self.a)

    def _prob_first_less_than_second(self, a1, b1, a2, b2):
        """Calculate P(X < Y) where X ~ U(a1,b1) and Y ~ U(a2,b2), assuming a1 <= a2.

        The probability is the area where x < y, divided by the total area.
        """
        assert a1 <= a2  # precondition

        if b1 <= a2:  # completely separate
            return 1.0

        # For each x in the overlap region, probability y > x is (b2-x)/(b2-a2)
        # No overlap region: probability is 1.0 for x < a2
        area = a2 - a1  # area where prob is 1.0
        if b1 > a2:
            # Add integral of (b2-x)/(b2-a2) from a2 to min(b1,b2)
            upper = min(b1, b2)
            area += (b2 * (upper - a2) - (upper**2 - a2**2) / 2) / (b2 - a2)

        return area / (b1 - a1)

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return Bernoulli(self.cdf(other))
        elif isinstance(other, Uniform):
            a1, b1 = self.a, self.b
            a2, b2 = other.a, other.b

            if a1 <= a2:
                prob = self._prob_first_less_than_second(a1, b1, a2, b2)
            else:
                # P(X < Y) = 1 - P(Y < X)
                prob = 1.0 - self._prob_first_less_than_second(a2, b2, a1, b1)

            return Bernoulli(prob)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Uniform(self.a + other, self.b + other)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Uniform(self.a * other, self.b * other)
        return NotImplemented
