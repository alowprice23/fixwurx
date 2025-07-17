"""
Input-validation utilities for the calculator application.
All functions now use precise error handling and stricter semantics.
"""

import math
from numbers import Number

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_float(value):
    """Safely convert *value* to float.

    Raises
    ------
    ValueError
        If *value* is not a valid finite real number.
    """
    # Reject booleans explicitly (bool is a subclass of int)
    if isinstance(value, bool):
        raise ValueError("Boolean values are not accepted as numbers.")

    num = float(value)

    if not math.isfinite(num):
        raise ValueError("Number must be finite (no NaN or Inf).")

    return num

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_numbers(*args):
    """Validate that all provided arguments are valid numbers.

    Raises
    ------
    ValueError
        If any value is not a valid finite real number.
    """
    for i, value in enumerate(args):
        try:
            _to_float(value)
        except (ValueError, TypeError) as e:
            ordinal = "first" if i == 0 else "second" if i == 1 else f"{i+1}th"
            raise ValueError(f"The {ordinal} operand is not a valid number: {str(e)}")

def is_number(value):
    """Return *True* if *value* represents a finite real number."""
    try:
        _to_float(value)
        return True
    except (ValueError, TypeError):
        return False


def is_positive(number):
    """Return *True* if *number* is strictly greater than 0."""
    try:
        return _to_float(number) > 0
    except (ValueError, TypeError):
        return False


def is_integer(value):
    """Return *True* if *value* represents an integer value (3, 3.0, "3", …)."""
    try:
        return _to_float(value).is_integer()
    except (ValueError, TypeError):
        return False


def is_in_range(value, min_val, max_val):
    """Return *True* if *value* lies inclusively between *min_val* and *max_val*."""
    try:
        num  = _to_float(value)
        low  = _to_float(min_val)
        high = _to_float(max_val)
    except (ValueError, TypeError):
        return False

    if low > high:  # user supplied bounds in reverse order
        low, high = high, low

    return low <= num <= high


_ALLOWED_OPERATIONS = {
    "add", "subtract", "multiply", "divide",  # long names
    "+", "-", "*", "/"                       # symbols
}


def validate_operation_inputs(a, b, operation):
    """Validate inputs for a calculator operation.

    Checks performed
    ----------------
    1. Operands *a* and *b* are numbers
    2. *operation* is one of the supported operations
    3. Division by zero is rejected
    """
    # 1. Supported operation? ------------------------------------------------
    if operation not in _ALLOWED_OPERATIONS:
        return False

    # 2. Are both operands numbers? -----------------------------------------
    if not (is_number(a) and is_number(b)):
        return False

    # 3. Division-by-zero protection ----------------------------------------
    if operation in {"divide", "/"} and _to_float(b) == 0:
        return False

    return True
