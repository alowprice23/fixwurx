from __future__ import annotations

"""
Advanced mathematical operations for the calculator application.
"""

import math
import random
from typing import Union

Number = Union[int, float]

def _is_real_number(value: object) -> bool:
    """Return ``True`` when *value* is an ``int`` or ``float`` **but not a bool**."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def square_root(x: Number) -> float:
    """Calculate the square root of a non-negative number."""
    if not _is_real_number(x):
        raise TypeError("Input must be a real number (int or float).")
    if x < 0:
        raise ValueError("Cannot compute the square root of a negative number.")
    return math.sqrt(float(x))

def factorial(n: int) -> int:
    """Return *n*! for non-negative integer *n*."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer.")
    if n < 0:
        raise ValueError("n must be non-negative.")
    # Use the highly-optimised C implementation
    return math.factorial(n)

def logarithm(x: Number, base: Number = 10) -> float:
    """Compute log_base(x)."""
    if not _is_real_number(x):
        raise TypeError("x must be a real number (int or float).")
    if not _is_real_number(base):
        raise TypeError("base must be a real number (int or float).")
    if x <= 0:
        raise ValueError("x must be positive for logarithm.")
    if base <= 0 or math.isclose(base, 1.0):
        raise ValueError("base must be positive and not equal to 1.")
    return math.log(float(x), float(base))

def sine(angle_degrees: Number) -> float:
    """Return the sine of *angle_degrees*."""
    if not _is_real_number(angle_degrees):
        raise TypeError("angle_degrees must be a real number (int or float).")
    angle_radians = math.radians(float(angle_degrees))
    return math.sin(angle_radians)

def random_operation(min_val: int, max_val: int) -> int:
    """Return a random integer *N* such that ``min_val ≤ N ≤ max_val``."""
    if (isinstance(min_val, bool) or isinstance(max_val, bool)
            or not isinstance(min_val, int) or not isinstance(max_val, int)):
        raise TypeError("min_val and max_val must be integers.")
    if min_val > max_val:
        raise ValueError("min_val cannot be greater than max_val.")
    return random.randint(min_val, max_val)
