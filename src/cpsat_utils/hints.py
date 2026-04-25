"""
Utilities for working with CP-SAT solution hints.

Provides functions to validate that hints are feasible and to complete
partial hints into full variable assignments. These are common operations
when warm-starting CP-SAT models from heuristic solutions or prior solves.

Usage:
    from cpsat_utils.hints import (
        assert_hint_feasible,
        complete_hint,
        hint_from_solution,
    )

    model = cp_model.CpModel()
    x = model.new_bool_var("x")
    model.add_hint(x, 1)
    assert_hint_feasible(model)  # raises if hints are infeasible
    complete_hint(model)         # fills in unhinted variables
    # After a solve, seed hints for the next iteration:
    hint_from_solution(model, solver)

When to modify:
    - If CP-SAT changes the hint validation API
    - To add hint analysis or diagnostics
"""

import logging
from collections.abc import Iterable

from ortools.sat.python import cp_model

logger = logging.getLogger(__name__)


_STATUS_NAMES_BY_CODE = {
    int(cp_model.UNKNOWN): "UNKNOWN",
    int(cp_model.MODEL_INVALID): "MODEL_INVALID",
    int(cp_model.FEASIBLE): "FEASIBLE",
    int(cp_model.INFEASIBLE): "INFEASIBLE",
    int(cp_model.OPTIMAL): "OPTIMAL",
}


def _backwards_compatible_status_name(status: object) -> str:
    """Return the name of a CpSolverStatus across ortools versions.

    ortools >= 9.11 exposes ``response_proto.status`` as a ``CpSolverStatus``
    enum (with a ``.name`` attribute). ortools 9.10 still returns a plain
    ``int``, so we map it back to the canonical name via a lookup table.
    """
    name = getattr(status, "name", None)
    if name is not None:
        return name
    code = int(status)  # type: ignore[arg-type]
    return _STATUS_NAMES_BY_CODE.get(code, f"status code {code}")


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


def hint_from_solution(
    model: cp_model.CpModel,
    solver: cp_model.CpSolver,
    variables: Iterable[cp_model.IntVar] | None = None,
    *,
    strict: bool = True,
) -> bool:
    """
    Replace the model's hints with values read from ``solver``.

    Intended for warm-starting a follow-up solve on the same model
    (LNS, lexicographic phases, incremental re-solves). When the solver
    has a usable solution, existing hints on ``model`` are cleared and
    replaced with the solver's values, so stale hints from previous
    iterations cannot leak through.

    Args:
        model: The CpModel to install hints on. Existing hints are cleared
            only if a solution is available.
        solver: A CpSolver that has just returned OPTIMAL or FEASIBLE on
            ``model``.
        variables: Variables to hint. Defaults to all variables in the
            model (mirrors :func:`complete_hint`'s behavior of walking
            every proto variable). Hinting only the decision variables is
            often sufficient in practice; CP-SAT can reconstruct the
            auxiliary ones.
        strict: If True (default), raise ``ValueError`` when the solver
            has no usable solution (status is not OPTIMAL or FEASIBLE,
            or ``solve()`` was never called). If False, leave existing
            hints untouched and return False — convenient inside
            iterative loops where an occasional time-out should not
            abort the run.

    Returns:
        True if hints were installed from the solver's solution, False if
        no solution was available and ``strict=False``.

    Raises:
        ValueError: If ``strict`` is True and the solver has no usable
            solution. Without a feasible solution, ``solver.value()``
            would silently write garbage hints.
    """
    try:
        status = solver.response_proto.status
        has_solution = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        status_label = _backwards_compatible_status_name(status)
    except RuntimeError:
        has_solution = False
        status_label = "solve() has not been called"

    if not has_solution:
        if strict:
            raise ValueError(
                "hint_from_solution requires a solver with status OPTIMAL "
                f"or FEASIBLE; got {status_label}."
            )
        logger.warning(
            "hint_from_solution: no usable solution (%s); hints unchanged.",
            status_label,
        )
        return False

    model.clear_hints()
    if variables is None:
        for i in range(len(model.proto.variables)):
            var = model.get_int_var_from_proto_index(i)
            model.add_hint(var, solver.value(var))
    else:
        for var in variables:
            model.add_hint(var, solver.value(var))
    return True
