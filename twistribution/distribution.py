from abc import abstractmethod, ABC
from typing import Any

from twistribution.constants import DEFAULT_EQUALITY_TOLERANCE


class Distribution(ABC):
    """Base class for all distributions."""

    def __init__(self, equality_tolerance: float = DEFAULT_EQUALITY_TOLERANCE):
        self.equality_tolerance = equality_tolerance

    def approximately_equal(self, a, b):
        return abs(a - b) < self.equality_tolerance

    @abstractmethod
    def parameters(self) -> tuple[Any, ...]:
        """All the distribution's parameter values, in a fixed order."""
        pass

    def __str__(self):
        return "{}({})".format(
            type(self).__name__, ", ".join(str(p) for p in self.parameters())
        )

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        for p1, p2 in zip(self.parameters(), other.parameters(), strict=True):
            if isinstance(p1, float) and isinstance(p2, float):
                if not self.approximately_equal(p1, p2):
                    return False
            elif p1 != p2:
                return False
        return True


class ContinuousDistribution(Distribution, ABC):
    """Base class for continuous distributions."""

    @abstractmethod
    def pdf(self, x: float) -> float:
        """Probability density function."""

    @abstractmethod
    def cdf(self, x: float) -> float:
        """Cumulative density function."""

    def __lt__(self, other):
        """
        Master comparator. All others are defined in terms of this.
        Default implementation that only works with scalars. Can be overridden to work against other distributions.
        """
        if isinstance(other, (float, int)):
            # avoid cyclic dependency by importing here
            from twistribution.bernoulli import Bernoulli

            return Bernoulli(self.cdf(other))
        return NotImplemented

    def __le__(self, other):
        return self.__lt__(other)

    def __gt__(self, other):
        # avoid cyclic dependency by importing here
        from twistribution.bernoulli import Bernoulli

        lt = self.__lt__(other)
        if lt is NotImplemented:
            return NotImplemented
        return Bernoulli(1 - lt.p)

    def __ge__(self, other):
        return self.__gt__(other)

    @abstractmethod
    def __add__(self, other):
        pass

    def __radd__(self, other):
        return self.__add__(other)

    @abstractmethod
    def __mul__(self, other):
        pass

    def __rmul__(self, other):
        return self.__mul__(other)


class DiscreteDistribution(Distribution, ABC):
    """Base class for discrete distributions."""

    @abstractmethod
    def __lt__(self, other):
        pass

    @abstractmethod
    def __le__(self, other):
        pass

    def __gt__(self, other):
        # avoid cyclic dependency by importing here
        from twistribution.bernoulli import Bernoulli

        le = self.__le__(other)
        if le is NotImplemented:
            return NotImplemented
        return Bernoulli(1 - le.p)

    def __ge__(self, other):
        # avoid cyclic dependency by importing here
        from twistribution.bernoulli import Bernoulli

        lt = self.__lt__(other)
        if lt is NotImplemented:
            return NotImplemented
        return Bernoulli(1 - lt.p)
