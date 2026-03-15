"""
Utilities for working with CP-SAT solution hints.

Provides functions to validate that hints are feasible and to complete
partial hints into full variable assignments. These are common operations
when warm-starting CP-SAT models from heuristic solutions or prior solves.

Usage:
    from cpsat_utils.hints import assert_hint_feasible, complete_hint

    model = cp_model.CpModel()
    x = model.new_bool_var("x")
    model.add_hint(x, 1)
    assert_hint_feasible(model)  # raises if hints are infeasible
    complete_hint(model)         # fills in unhinted variables

When to modify:
    - If CP-SAT changes the hint validation API
    - To add hint analysis or diagnostics
"""

import logging

from ortools.sat.python import cp_model

logger = logging.getLogger(__name__)


def assert_hint_feasible(
    model: cp_model.CpModel,
    time_limit: float = 10.0,
) -> None:
    """
    Assert that the current hints on the model are feasible.

    Solves the model with all hinted variables fixed to their hinted values.
    Raises AssertionError if the resulting model is infeasible.

    Args:
        model: A CpModel with hints already set via ``model.add_hint()``.
        time_limit: Maximum solve time in seconds.

    Raises:
        AssertionError: If the hints lead to an infeasible model.
    """
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.fix_variables_to_their_hinted_value = True
    status = solver.solve(model)
    assert status != cp_model.INFEASIBLE, (
        "Hints are infeasible: fixing hinted variables leads to "
        f"status {solver.status_name(status)}."
    )
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
        "Could not verify hint feasibility within time limit: "
        f"status {solver.status_name(status)}. "
        "Try increasing the time_limit."
    )


def complete_hint(
    model: cp_model.CpModel,
    time_limit: float = 10.0,
) -> bool:
    """
    Complete partial hints into a full variable assignment.

    CP-SAT only benefits from complete hints (all variables hinted).
    This function does a quick solve with hinted variables fixed,
    then sets hints for all remaining variables based on the solution.

    Args:
        model: A CpModel with partial hints set via ``model.add_hint()``.
        time_limit: Maximum solve time in seconds for hint completion.

    Returns:
        True if hints were successfully completed, False otherwise.
        On failure, the model's hints are left unchanged.
    """
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.fix_variables_to_their_hinted_value = True
    status = solver.solve(model)
    logger.info(
        "Hint completion solve returned status: %s",
        solver.status_name(status),
    )
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.warning(
            "Unable to complete hint: status %s. Hints are left unchanged.",
            solver.status_name(status),
        )
        return False

    model.clear_hints()
    for i in range(len(model.proto.variables)):
        var = model.get_int_var_from_proto_index(i)
        model.add_hint(var, solver.value(var))
    logger.info("Hints successfully completed.")
    return True
