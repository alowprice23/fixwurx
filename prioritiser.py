"""
goal/prioritiser.py
───────────────────
Enhanced scoring system that converts an *incoming* `BugTicket` (see `core.scheduler`)
into a priority score suitable for Python's `sorted(..., reverse=True)` call.

Mathematical background
───────────────────────
The enhanced priority formula combines multiple factors:

priority = w₁·S_norm + w₂·A_norm + w₃·E_norm + w₄·H_norm + w₅·D_norm

Where:
  S_norm = severity / max_severity              ∈ [0, 1]  (severity)
  A_norm = min(1, age / AGE_MAX)                ∈ [0, 1]  (age)
  E_norm = entropy / MAX_ENTROPY                ∈ [0, 1]  (complexity)
  H_norm = (1 + success_rate) / 2               ∈ [0, 1]  (historical success)
  D_norm = dependency_factor                    ∈ [0, 1]  (dependency impact)

To guarantee **starvation-freedom** we must ensure that, for sufficiently old
tickets, the age term can dominate any combination of other terms:

      w₂ > w₁ + w₃ + w₄ + w₅

WEIGHTS are configurable and can be adjusted dynamically by the planner agent
based on learning from previous fixes.

This enhanced prioritization integrates with the planner for intelligent
scheduling based on past outcomes and solution path effectiveness.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from core.scheduler import BugTicket

# ---------------------------------------------------------------------------
# Tunable constants (can be overridden by optimiser if desired)
# ---------------------------------------------------------------------------
MAX_SEVERITY: int = 5
MAX_ENTROPY: float = 5.0     # maximum expected entropy value

# Default weights - these can be adjusted dynamically
WEIGHTS = {
    "severity": 0.25,        # w₁: weight on severity
    "age": 0.35,             # w₂: weight on age
    "entropy": 0.15,         # w₃: weight on complexity
    "history": 0.15,         # w₄: weight on historical success
    "dependency": 0.10,      # w₅: weight on dependency impact
}

AGE_MAX: float = 45.0        # seconds until age term saturates to 1.0

# Path to historical success data
HISTORY_PATH = Path(".triangulum") / "fix_history.json"
HISTORY_PATH.parent.mkdir(exist_ok=True)

# Cache for history and dependency data to avoid repeated calculations
_history_cache: Dict[str, float] = {}
_dependency_cache: Dict[str, float] = {}
_entropy_cache: Dict[str, float] = {}


# ---------------------------------------------------------------------------
# Helper functions for enhanced prioritization
# ---------------------------------------------------------------------------
def _load_history() -> Dict[str, Dict[str, Union[int, float]]]:
    """Load historical success rate data from disk."""
    if not HISTORY_PATH.exists():
        return {}
    
    try:
        with open(HISTORY_PATH, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def _save_history(history: Dict[str, Dict[str, Union[int, float]]]) -> None:
    """Save historical success rate data to disk."""
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)


def _get_success_rate(bug_type: str) -> float:
    """Get historical success rate for this type of bug."""
    if bug_type in _history_cache:
        return _history_cache[bug_type]
    
    history = _load_history()
    bug_history = history.get(bug_type, {})
    
    attempts = bug_history.get("attempts", 0)
    successes = bug_history.get("successes", 0)
    
    # Default to 0.5 if no history (neither positive nor negative bias)
    if attempts == 0:
        rate = 0.5
    else:
        rate = successes / attempts
    
    _history_cache[bug_type] = rate
    return rate


def _get_dependency_factor(ticket: BugTicket) -> float:
    """
    Calculate a dependency factor based on how many other bugs depend on this one.
    
    Higher values indicate this bug blocks more other bugs.
    """
    # This would typically connect to a dependency graph system
    # For now, we use a simple approach based on bug tags
    if not hasattr(ticket, "tags") or not ticket.tags:
        return 0.5  # Default mid-value
    
    # Check if this bug is marked as a blocker
    if "blocker" in ticket.tags:
        return 1.0
    elif "dependency" in ticket.tags:
        return 0.8
    else:
        return 0.3


def _get_entropy_value(ticket: BugTicket) -> float:
    """Calculate entropy (complexity) for this bug."""
    if hasattr(ticket, "entropy") and ticket.entropy is not None:
        return min(ticket.entropy, MAX_ENTROPY) / MAX_ENTROPY
    
    # If ticket doesn't have entropy, estimate it from description length and severity
    if hasattr(ticket, "description") and ticket.description:
        desc_length = len(ticket.description)
        estimated_entropy = min(math.log(1 + desc_length) / 10, 1.0)
        return (estimated_entropy + (ticket.severity / MAX_SEVERITY)) / 2
    
    # Default value if no information available
    return 0.5


def _update_success_history(bug_type: str, success: bool) -> None:
    """Update historical success rate for this type of bug."""
    history = _load_history()
    
    if bug_type not in history:
        history[bug_type] = {"attempts": 0, "successes": 0}
    
    history[bug_type]["attempts"] = history[bug_type].get("attempts", 0) + 1
    if success:
        history[bug_type]["successes"] = history[bug_type].get("successes", 0) + 1
    
    _save_history(history)
    
    # Update cache
    _history_cache[bug_type] = history[bug_type]["successes"] / history[bug_type]["attempts"]


# ---------------------------------------------------------------------------
# Public scoring function
# ---------------------------------------------------------------------------
def score(ticket: BugTicket) -> float:
    """
    Return an enhanced priority score ∈ [0, 1+ε].  Higher = more urgent.
    Considers multiple factors including severity, age, complexity, 
    historical success, and dependencies.

    The scheduler sorts with `reverse=True`, so larger comes first.
    """
    now = time.time()

    # 1. Normalized severity
    s_norm = min(ticket.severity, MAX_SEVERITY) / MAX_SEVERITY

    # 2. Normalized age
    age_sec = max(0.0, now - ticket.arrival_ts)
    a_norm = min(1.0, age_sec / AGE_MAX)

    # 3. Get bug type (fallback to "unknown" if not available)
    bug_type = getattr(ticket, "bug_type", "unknown")
    
    # 4. Entropy/complexity normalization
    e_norm = _get_entropy_value(ticket)
    
    # 5. Historical success normalization
    h_norm = _get_success_rate(bug_type)
    
    # 6. Dependency factor
    d_norm = _get_dependency_factor(ticket)

    # 7. Weighted combination of all factors
    return (
        WEIGHTS["severity"] * s_norm +
        WEIGHTS["age"] * a_norm +
        WEIGHTS["entropy"] * e_norm +
        WEIGHTS["history"] * h_norm +
        WEIGHTS["dependency"] * d_norm
    )


def update_weights(new_weights: Dict[str, float]) -> None:
    """
    Update the prioritization weights. This can be called by the planner
    to dynamically adjust prioritization strategy.
    """
    global WEIGHTS
    
    # Validate weights
    total = sum(new_weights.values())
    if abs(total - 1.0) > 0.001:
        # Normalize weights to sum to 1
        for key in new_weights:
            new_weights[key] = new_weights[key] / total
    
    # Ensure age weight satisfies the starvation-freedom constraint
    other_weights_sum = sum(w for k, w in new_weights.items() if k != "age")
    if new_weights["age"] <= other_weights_sum:
        # Adjust weights to satisfy constraint
        new_weights["age"] = other_weights_sum + 0.01
        # Re-normalize
        total = sum(new_weights.values())
        for key in new_weights:
            new_weights[key] = new_weights[key] / total
    
    # Update weights
    for key, value in new_weights.items():
        if key in WEIGHTS:
            WEIGHTS[key] = value


def record_outcome(bug_type: str, success: bool) -> None:
    """
    Record the outcome of a fix attempt. This helps build historical
    success rates for different bug types.
    """
    _update_success_history(bug_type, success)


# ---------------------------------------------------------------------------
# Starvation-freedom sanity check  (defensive; runs once on import)
# ---------------------------------------------------------------------------
other_weights_sum = sum(w for k, w in WEIGHTS.items() if k != "age")
if WEIGHTS["age"] <= other_weights_sum:
    raise RuntimeError(
        "Priority weights violate starvation-freedom constraint: "
        f"age weight must be greater than sum of other weights ({other_weights_sum:.3f})"
    )


# ---------------------------------------------------------------------------
# Convenience helper for CLI / debugging
# ---------------------------------------------------------------------------
def explain(ticket: BugTicket) -> str:
    """Return human-readable breakdown for dashboards."""
    s_norm = min(ticket.severity, MAX_SEVERITY) / MAX_SEVERITY
    age = time.time() - ticket.arrival_ts
    a_norm = min(1, age / AGE_MAX)
    
    bug_type = getattr(ticket, "bug_type", "unknown")
    e_norm = _get_entropy_value(ticket)
    h_norm = _get_success_rate(bug_type)
    d_norm = _get_dependency_factor(ticket)
    
    w = WEIGHTS
    
    return (
        f"priority={score(ticket):.3f}\n"
        f"  severity:   {ticket.severity}→{s_norm:.2f} × {w['severity']:.2f} = {w['severity'] * s_norm:.2f}\n"
        f"  age:        {age:.1f}s→{a_norm:.2f} × {w['age']:.2f} = {w['age'] * a_norm:.2f}\n"
        f"  entropy:    {e_norm:.2f} × {w['entropy']:.2f} = {w['entropy'] * e_norm:.2f}\n"
        f"  history:    {h_norm:.2f} × {w['history']:.2f} = {w['history'] * h_norm:.2f}\n"
        f"  dependency: {d_norm:.2f} × {w['dependency']:.2f} = {w['dependency'] * d_norm:.2f}"
    )


@dataclass
class FactorBreakdown:
    """Detailed breakdown of priority score factors for advanced analysis."""
    severity: float
    age: float
    entropy: float
    history: float
    dependency: float
    severity_contrib: float
    age_contrib: float
    entropy_contrib: float
    history_contrib: float
    dependency_contrib: float
    total_score: float


def get_detailed_breakdown(ticket: BugTicket) -> FactorBreakdown:
    """Get detailed breakdown of all contributing factors for analytics."""
    s_norm = min(ticket.severity, MAX_SEVERITY) / MAX_SEVERITY
    age = time.time() - ticket.arrival_ts
    a_norm = min(1, age / AGE_MAX)
    
    bug_type = getattr(ticket, "bug_type", "unknown")
    e_norm = _get_entropy_value(ticket)
    h_norm = _get_success_rate(bug_type)
    d_norm = _get_dependency_factor(ticket)
    
    w = WEIGHTS
    
    return FactorBreakdown(
        severity=s_norm,
        age=a_norm,
        entropy=e_norm,
        history=h_norm,
        dependency=d_norm,
        severity_contrib=w["severity"] * s_norm,
        age_contrib=w["age"] * a_norm,
        entropy_contrib=w["entropy"] * e_norm,
        history_contrib=w["history"] * h_norm,
        dependency_contrib=w["dependency"] * d_norm,
        total_score=score(ticket)
    )


# ---------------------------------------------------------------------------
# Type alias for mypy users who call `score` in key=…
# ---------------------------------------------------------------------------
ScoreType = Union[int, float]
