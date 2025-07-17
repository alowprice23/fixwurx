from __future__ import annotations
"""
Basic arithmetic operations for the calculator application.
Now contains stricter type validation, configurable near-zero handling,
and reduced code duplication.
"""

import math
from numbers import Real
from typing import Final

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Number = Real  # Accepts int, float, decimal.Decimal, fractions.Fraction, etc.

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
_NEAR_ZERO_TOLERANCE: Final[float] = 1e-12  # configurable absolute tolerance

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_numbers(*args: Number) -> None:
    """Ensure every argument is a real number (but *not* bool)."""
    for arg in args:
        # Reject bool first – bool is a subclass of int
        if isinstance(arg, bool):
            raise TypeError("Operands must be numeric (int/float/etc.), bool is not allowed.")
        if not isinstance(arg, Real):
            raise TypeError(
                f"Operands must be real numbers (int, float, Decimal, Fraction, ...); got {type(arg).__name__}."
            )


def _validate_non_zero_divisor(b: Number) -> None:
    """Ensure *b* is not numerically zero (within tolerance)."""
    if math.isclose(b, 0.0, abs_tol=_NEAR_ZERO_TOLERANCE):
        raise ZeroDivisionError("Division or modulus by (near) zero is undefined.")

# ---------------------------------------------------------------------------
# Arithmetic primitives
# ---------------------------------------------------------------------------

def add(a: Number, b: Number) -> Number:
    """Return *a + b*."""
    _validate_numbers(a, b)
    return a + b


def subtract(a: Number, b: Number) -> Number:
    """Return *a - b*."""
    _validate_numbers(a, b)
    return a - b


def multiply(a: Number, b: Number) -> Number:
    """Return *a * b*."""
    _validate_numbers(a, b)
    return a * b


def divide(a: Number, b: Number) -> Number:
    """Return *a / b*.

    Raises
    ------
    ZeroDivisionError
        If *b* is numerically zero (|b| < ``_NEAR_ZERO_TOLERANCE``).
    """
    _validate_numbers(a, b)
    _validate_non_zero_divisor(b)
    return a / b


def power(a: Number, b: Number) -> Number:
    """Return *a* raised to *b*."""
    _validate_numbers(a, b)
    return a ** b


def modulus(a: Number, b: Number) -> Number:
    """Return *a % b*.

    Raises
    ------
    ZeroDivisionError
        If *b* is numerically zero (|b| < ``_NEAR_ZERO_TOLERANCE``).
    """
    _validate_numbers(a, b)
    _validate_non_zero_divisor(b)
    return a % b
