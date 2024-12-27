from collections import defaultdict
from typing import Any

from twistribution.bernoulli import Bernoulli
from twistribution.distribution import DiscreteDistribution


class Discrete(DiscreteDistribution):
    """
    Numerical values and associated probabilities.
    """

    def __init__(
        self, probabilities: dict[float | int, float], equality_tolerance=1e-8
    ):
        if not probabilities:
            raise ValueError("Probabilities cannot be empty")

        key_list = list(probabilities.keys())
        for k1, k2 in zip(key_list, key_list[1:]):
            if k1 > k2:
                raise ValueError("Keys must be in ascending order")

        sum_of_probabilities = sum(probabilities.values())
        if abs(1 - sum_of_probabilities) > 1e-8:
            raise ValueError("Probability values must sum to 1")

        self.probabilities = probabilities
        self.equality_tolerance = equality_tolerance

    def parameters(self) -> tuple[Any, ...]:
        return self.probabilities, self.equality_tolerance

    def __eq__(self, other):
        if not isinstance(other, Discrete):
            return NotImplemented

        if set(self.probabilities.keys()) != set(other.probabilities.keys()):
            return False

        for key in self.probabilities:
            if (
                abs(self.probabilities[key] - other.probabilities[key])
                > self.equality_tolerance
            ):
                return False

        return True

    def mean(self) -> float:
        return sum(value * prob for value, prob in self.probabilities.items())

    def median(self, tie_margin: float = 1e-6) -> float:
        cumulative = 0
        previous_cumulative = 0
        previous_value = None
        for value, prob in self.probabilities.items():
            cumulative += prob
            if cumulative > 0.5 + tie_margin:
                if (0.5 - previous_cumulative) < 2 * tie_margin:
                    # effectively a tie
                    return (previous_value + value) / 2
                return value
            previous_value = value
            previous_cumulative = cumulative
        raise RuntimeError("Median calculation failed: probabilities may be malformed")

    def __add__(self, other):
        if isinstance(other, (float, int)):
            return Discrete({k + other: v for (k, v) in self.probabilities.items()})
        elif isinstance(other, Discrete):
            new_probabilities = defaultdict(lambda: 0.0)
            for v1, p1 in self.probabilities.items():
                for v2, p2 in other.probabilities.items():
                    new_probabilities[v1 + v2] += p1 * p2
            sorted_dict = dict(sorted(new_probabilities.items()))
            return Discrete(sorted_dict)
        raise TypeError(f"Unsupported operation between Discrete and {type(other)}")

    def __radd__(self, other):
        return self.__add__(other)

    def __gt__(self, other):
        if isinstance(other, (float, int)):
            if other < next(iter(self.probabilities.keys())):
                return Bernoulli(1.0)

            cumulative = 0.0
            for k, v in self.probabilities.items():
                if k > other:
                    return Bernoulli(1.0 - cumulative)
                cumulative += v

            return Bernoulli(0.0)

    def __ge__(self, other):
        if isinstance(other, (float, int)):
            if other < next(iter(self.probabilities.keys())):
                return Bernoulli(1.0)

            cumulative = 0.0
            for k, v in self.probabilities.items():
                if k >= other:
                    return Bernoulli(1.0 - cumulative)
                cumulative += v

            return Bernoulli(0.0)
