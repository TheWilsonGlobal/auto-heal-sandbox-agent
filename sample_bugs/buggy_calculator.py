"""Buggy calculator implementation with edge-case and off-by-one flaws."""

from typing import List, Union


class Calculator:
    """A mathematical operations calculator with intentional bugs."""

    def add(self, a: float, b: float) -> float:
        return a + b

    def subtract(self, a: float, b: float) -> float:
        return a - b

    def multiply(self, a: float, b: float) -> float:
        return a * b

    def divide(self, a: float, b: float) -> float:
        # BUG: Fails to raise ValueError on division by zero
        # Causes ZeroDivisionError instead of handled ValueError
        return a / b

    def power(self, base: float, exponent: float) -> float:
        return base ** exponent

    def average(self, numbers: List[Union[int, float]]) -> float:
        # BUG 1: Does not handle empty list gracefully (raises ZeroDivisionError or crash)
        # BUG 2: Off-by-one error in denominator: dividing by (len - 1) instead of len
        if not numbers:
            return 0.0
        return sum(numbers) / (len(numbers) - 1)

    def factorial(self, n: int) -> int:
        # BUG: Does not validate negative integers, causing infinite recursion
        if n == 0 or n == 1:
            return 1
        return n * self.factorial(n - 1)
