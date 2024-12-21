from twistribution.bernoulli import Bernoulli


class ContinuousDistribution:
    def __ge__(self, other):
        return self.__gt__(other)

    def __lt__(self, other):
        ge = self.__ge__(other)
        return Bernoulli(1 - ge.p)

    def __le__(self, other):
        return self.__lt__(other)
