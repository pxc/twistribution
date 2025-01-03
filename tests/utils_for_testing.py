import random


def close(a: float, b: float, tolerance: float = 1e-8) -> bool:
    return abs(a - b) < tolerance


def estimate_probability(samples=1_000_000):
    """Decorator that estimates probability by random sampling"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            count = 0
            random.seed(42)
            for _ in range(samples):
                if func(*args, **kwargs):
                    count += 1
            return count / samples

        return wrapper

    return decorator
