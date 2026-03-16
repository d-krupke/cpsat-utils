"""
Utilities for Google's ortools CP-SAT solver.

Provides context managers and assertion helpers for testing CP-SAT models,
hint management utilities, and model import/export.
"""

# When bumping __version__, also update CHANGELOG.md with the new release entry.
__version__ = "0.4.1"

from cpsat_utils.hints import assert_hint_feasible, complete_hint
from cpsat_utils.io import export_model, import_model
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
    "assert_hint_feasible",
    "assert_infeasible",
    "assert_objective",
    "assert_optimal",
    "complete_hint",
    "export_model",
    "import_model",
    "solve",
]
