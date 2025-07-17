from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional


class CalculationHistory:
    """Class to store and manage calculation history with a fixed maximum size."""

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(self, max_size: int = 10):
        """Initialise an empty history.

        Parameters
        ----------
        max_size : int, optional
            The maximum number of calculations to keep. Must be positive.
        """
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer")

        self._max_size: int = max_size  # make read-only via a property
        self._history: List[Dict[str, Any]] = []  # underscore to signal private

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def max_size(self) -> int:
        """Read-only access to the maximum history size."""
        return self._max_size

    def add_calculation(self, operation: str, a: Any, b: Any, result: Any) -> None:
        """Add a calculation to the history, automatically trimming if needed."""
        if not isinstance(operation, str) or not operation:
            raise ValueError("'operation' must be a non-empty string")

        # Store *deep* copies so the history is immune to later mutations
        calculation: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc),
            "operation": operation,
            "a": copy.deepcopy(a),
            "b": copy.deepcopy(b),
            "result": copy.deepcopy(result),
        }

        self._history.append(calculation)

        # Trim to `max_size` – remove oldest entries first
        if len(self._history) > self._max_size:
            excess = len(self._history) - self._max_size
            del self._history[:excess]

    def get_last_calculation(self) -> Optional[Dict[str, Any]]:
        """Return the most recent calculation or ``None`` if history is empty."""
        if not self._history:
            return None
        # Return a deep copy to prevent external mutations
        return copy.deepcopy(self._history[-1])

    def clear_history(self) -> None:
        """Remove all stored calculations."""
        self._history.clear()

    def get_all_calculations(self) -> List[Dict[str, Any]]:
        """Return *deep* copies of all calculations in chronological order."""
        return copy.deepcopy(self._history)

    def get_calculations_by_operation(self, operation: str) -> List[Dict[str, Any]]:
        """Return calculations whose `operation` field matches the supplied value."""
        if not isinstance(operation, str) or not operation:
            raise ValueError("'operation' must be a non-empty string")
        return [copy.deepcopy(calc) for calc in self._history if calc["operation"] == operation]

    # ------------------------------------------------------------------
    # Dunder methods for convenience
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # allows len(history)
        return len(self._history)

    def __iter__(self) -> Iterator[Dict[str, Any]]:  # iterate over calculations safely
        # iterate over a snapshot copy so callers can't mutate internal state
        return iter(self.get_all_calculations())
