"""Pytest suite specifying expected behavior of Calculator."""

import pytest

# Supports execution both in the sandbox as solution.py and locally
try:
    from solution import Calculator
except ImportError:
    from buggy_calculator import Calculator


def test_basic_arithmetic():
    calc = Calculator()
    assert calc.add(2, 3) == 5
    assert calc.subtract(10, 4) == 6
    assert calc.multiply(3, 7) == 21
    assert calc.divide(10, 2) == 5.0
    assert calc.power(2, 3) == 8


def test_divide_by_zero():
    calc = Calculator()
    # Expects ValueError when dividing by zero
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        calc.divide(10, 0)


def test_average():
    calc = Calculator()
    assert calc.average([10, 20, 30]) == 20.0
    assert calc.average([5]) == 5.0
    assert calc.average([]) == 0.0


def test_factorial():
    calc = Calculator()
    assert calc.factorial(0) == 1
    assert calc.factorial(1) == 1
    assert calc.factorial(5) == 120
    with pytest.raises(ValueError, match="Factorial is not defined for negative numbers"):
        calc.factorial(-3)
