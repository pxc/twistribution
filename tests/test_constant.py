from twistribution.constant import Constant


def test_constant():
    c = Constant(1)
    assert c.pmf(1) == 1, f"{c.pmf(1)}"
    assert c.pmf(2) == 0, f"{c.pmf(2)}"
    assert c.cdf(1) == 1, f"{c.cdf(1)}"
    assert c.cdf(2) == 1, f"{c.cdf(2)}"
    assert c.mean == 1, f"{c.mean}"
    assert c.variance == 0, f"{c.variance}"


def test_constant_lt():
    c1 = Constant(1)
    c2 = Constant(2)
    assert (c1 < c2) == Constant(1), f"{c1 < c2}"


def test_constant_lt_scalar():
    c = Constant(1)
    assert (c < 2) == Constant(1), f"{c < 2}"
    assert (c < 1.5) == Constant(1), f"{c < 1.5}"
    assert (c < 1) == Constant(0), f"{c < 1}"
    assert (c < 0.5) == Constant(0), f"{c < 0.5}"


def test_constant_le_scalar():
    c = Constant(1)
    assert (c <= 2) == Constant(1), f"{c <= 2}"
    assert (c <= 1.5) == Constant(1), f"{c <= 1.5}"
    assert (c <= 1) == Constant(1), f"{c <= 1}"
    assert (c <= 0) == Constant(0), f"{c <= 0}"


def test_constant_float():
    c = Constant(1)
    assert float(c) == 1, f"{float(c)}"


def test_constant_add():
    c1 = Constant(1)
    c2 = Constant(2)
    assert (c1 + c2) == Constant(3), f"{c1 + c2}"


def test_constant_add_scalar():
    c = Constant(1)
    assert (c + 2) == Constant(3), f"{c + 2}"
    assert (c + 1.5) == Constant(2.5), f"{c + 1.5}"
    assert (c + 1) == Constant(2), f"{c + 1}"
    assert (c + 0.5) == Constant(1.5), f"{c + 0.5}"


def test_constant_sub():
    c1 = Constant(1)
    c2 = Constant(2)
    assert (c1 - c2) == Constant(-1), f"{c1 - c2}"


def test_constant_sub_scalar():
    c = Constant(1)
    assert (c - 2) == Constant(-1), f"{c - 2}"
    assert (c - 1.5) == Constant(-0.5), f"{c - 1.5}"
    assert (c - 1) == Constant(0), f"{c - 1}"
    assert (c - 0.5) == Constant(0.5), f"{c - 0.5}"


def test_constant_mul():
    c1 = Constant(2)
    c2 = Constant(3)
    assert (c1 * c2) == Constant(6), f"{c1 * c2}"


def test_constant_mul_scalar():
    c = Constant(2)
    assert (c * 3) == Constant(6), f"{c * 3}"
    assert (c * 1.5) == Constant(3), f"{c * 1.5}"
    assert (c * 1) == Constant(2), f"{c * 1}"
    assert (c * 0.5) == Constant(1), f"{c * 0.5}"


def test_constant_truediv():
    c1 = Constant(2)
    c2 = Constant(3)
    assert (c1 / c2) == Constant(2 / 3), f"{c1 / c2}"


def test_constant_truediv_scalar():
    c = Constant(2)
    assert (c / 3) == Constant(2 / 3), f"{c / 3}"
    assert (c / 1.5) == Constant(2 / 1.5), f"{c / 1.5}"
    assert (c / 1) == Constant(2), f"{c / 1}"
    assert (c / 0.5) == Constant(2 / 0.5), f"{c / 0.5}"
