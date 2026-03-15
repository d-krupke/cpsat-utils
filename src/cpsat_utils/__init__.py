"""
Utilities for Google's ortools CP-SAT solver.

Provides context managers and assertion helpers for testing CP-SAT models.
"""

__version__ = "0.2.0"

from cpsat_utils.testing import (
    AssertModelFeasible,
    AssertModelInfeasible,
    AssertObjectiveValue,
    AssertOptimalWithinTime,
    assert_feasible,
    assert_infeasible,
    assert_objective,
    assert_optimal,
    solve,
)

__all__ = [
    "AssertModelFeasible",
    "AssertModelInfeasible",
    "AssertObjectiveValue",
    "AssertOptimalWithinTime",
    "assert_feasible",
    "assert_infeasible",
    "assert_objective",
    "assert_optimal",
    "solve",
]
