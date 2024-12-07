"""Collection of the core mathematical operators used throughout the code base."""

import math

from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """Multiplies two numbers."""
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negates a number."""
    return -x


def lt(x: float, y: float) -> float:
    """Checks if one number is less than another."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if two numbers are equal."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Maximum of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value."""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Rectified linear unit (ReLU) function."""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Natural logarithm function."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Exponential function."""
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Derivative of natural logarithm multiplied by a second argument."""
    return y / (x + EPS)


def inv(x: float) -> float:
    """Inverse function"""
    return 1.0 / x


def inv_back(x: float, y: float) -> float:
    """Derivative of inverse function multiplied by a second argument."""
    return -(1.0 / x**2) * y


def relu_back(x: float, y: float) -> float:
    """Derivative of rectified linear unit (ReLU) function multiplied by a second argument."""
    return y if x > 0 else 0.0


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Apply a function to each element in a list."""

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Apply a function to pairs of elements combined from two lists."""

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], initial: float
) -> Callable[[Iterable[float]], float]:
    """Reduce an iterable to a single value using a function."""

    def _reduce(ls: Iterable[float]) -> float:
        ret = initial
        for x in ls:
            ret = fn(ret, x)
        return ret

    return _reduce


def negList(list: Iterable[float]) -> Iterable[float]:
    """Negate each element in a list."""
    return map(neg)(list)


def addLists(list1: Iterable[float], list2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists."""
    return zipWith(add)(list1, list2)


def sum(list: Iterable[float]) -> float:
    """Sum all elements in a list."""
    return reduce(add, 0.0)(list)


def prod(list: Iterable[float]) -> float:
    """Multiply all elements in a list."""
    return reduce(mul, 1.0)(list)
