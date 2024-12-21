class Bernoulli:
    def __init__(self, p: float):
        if not (0 <= p <= 1):
            raise ValueError("Probability p must be in the range [0, 1]")
        self.p = p

    def __str__(self):
        return f"Bernoulli({self.p})"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, Bernoulli):
            return NotImplemented
        return self.p == other.p

    def mean(self):
        return self.p

    def variance(self):
        return self.p * (1 - self.p)
