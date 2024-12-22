def close(a: float, b: float, tolerance: float = 1e-8) -> bool:
    return abs(a - b) < tolerance
