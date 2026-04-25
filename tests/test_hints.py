"""
Tests for cpsat_utils.hints (assert_hint_feasible, complete_hint,
hint_from_solution).
"""

import pytest
from ortools.sat.python import cp_model

from cpsat_utils.hints import (
    assert_hint_feasible,
    complete_hint,
    hint_from_solution,
)


class TestAssertHintFeasible:
    def test_feasible_hint_passes(self):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        y = model.new_bool_var("y")
        model.add(x + y == 1)
        model.add_hint(x, 1)
        model.add_hint(y, 0)
        assert_hint_feasible(model)

    def test_infeasible_hint_raises(self):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        y = model.new_bool_var("y")
        model.add(x + y == 1)
        model.add_hint(x, 1)
        model.add_hint(y, 1)  # contradicts x + y == 1
        with pytest.raises(AssertionError, match="infeasible"):
            assert_hint_feasible(model)

    def test_partial_hint_feasible(self):
        """A partial hint that doesn't conflict should pass."""
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        model.new_bool_var("y")  # unhinted
        model.add(x <= 1)
        model.add_hint(x, 0)
        assert_hint_feasible(model)

    def test_hint_with_objective(self):
        """Hints should be checked for feasibility, not optimality."""
        model = cp_model.CpModel()
        x = model.new_int_var(0, 10, "x")
        model.add_hint(x, 5)  # not optimal but feasible
        model.minimize(x)
        assert_hint_feasible(model)


class TestCompleteHint:
    def test_completes_partial_hint(self):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        y = model.new_bool_var("y")
        model.add(x + y == 1)
        model.add_hint(x, 1)
        # y is unhinted

        result = complete_hint(model)
        assert result is True

        # After completion, all variables should have hints.
        # Verify by checking the proto hint fields.
        hints = dict(
            zip(
                model.proto.solution_hint.vars,
                model.proto.solution_hint.values,
                strict=True,
            )
        )
        assert len(hints) == 2

    def test_returns_false_on_infeasible_hint(self):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        y = model.new_bool_var("y")
        model.add(x + y == 1)
        model.add_hint(x, 1)
        model.add_hint(y, 1)  # contradicts x + y == 1

        result = complete_hint(model)
        assert result is False

    def test_completed_hint_is_feasible(self):
        """After completing, fixing to hints should still be feasible."""
        model = cp_model.CpModel()
        x = model.new_int_var(0, 5, "x")
        y = model.new_int_var(0, 5, "y")
        model.add(x + y == 4)
        model.add_hint(x, 2)

        complete_hint(model)
        assert_hint_feasible(model)

    def test_no_hints_still_works(self):
        """Completing with no hints set should still succeed."""
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        y = model.new_bool_var("y")
        model.add(x + y <= 1)

        result = complete_hint(model)
        assert result is True


class TestHintFromSolution:
    """Tests for warm-starting follow-up solves with the previous solution."""

    def test_raises_on_unknown_status(self):
        """Solver that has not been run yet has no solution to read."""
        model = cp_model.CpModel()
        model.new_bool_var("x")
        solver = cp_model.CpSolver()
        with pytest.raises(ValueError, match="OPTIMAL or FEASIBLE"):
            hint_from_solution(model, solver)

    def test_raises_on_infeasible_status(self):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        y = model.new_bool_var("y")
        model.add(x + y == 1)
        model.add(x + y == 2)  # infeasible
        solver = cp_model.CpSolver()
        solver.solve(model)
        with pytest.raises(ValueError, match="OPTIMAL or FEASIBLE"):
            hint_from_solution(model, solver)

    def test_clears_existing_hints(self):
        """A stale hint from a previous iteration must not survive."""
        model = cp_model.CpModel()
        x = model.new_int_var(0, 10, "x")
        y = model.new_int_var(0, 10, "y")
        model.add(x + y == 5)

        # Stale hints from a "previous iteration" — both will be wrong on
        # the next solve since the constraint forces x + y == 5.
        model.add_hint(x, 9)
        model.add_hint(y, 9)

        solver = cp_model.CpSolver()
        solver.solve(model)
        hint_from_solution(model, solver)

        hints = dict(
            zip(
                model.proto.solution_hint.vars,
                model.proto.solution_hint.values,
                strict=True,
            )
        )
        # Old (9, 9) must be gone; new hint values must satisfy x + y == 5.
        assert hints[x.index] + hints[y.index] == 5

    def test_default_hints_all_proto_variables(self):
        model = cp_model.CpModel()
        x = model.new_int_var(0, 5, "x")
        y = model.new_int_var(0, 5, "y")
        z = model.new_int_var(0, 5, "z")
        model.add(x + y + z == 6)
        solver = cp_model.CpSolver()
        solver.solve(model)

        hint_from_solution(model, solver)

        hint_indices = set(model.proto.solution_hint.vars)
        assert hint_indices == {x.index, y.index, z.index}

    def test_explicit_subset(self):
        """Only listed variables are hinted; others remain unhinted."""
        model = cp_model.CpModel()
        x = model.new_int_var(0, 5, "x")
        y = model.new_int_var(0, 5, "y")
        z = model.new_int_var(0, 5, "z")
        model.add(x + y + z == 6)
        solver = cp_model.CpSolver()
        solver.solve(model)

        hint_from_solution(model, solver, variables=[x, y])

        hint_indices = set(model.proto.solution_hint.vars)
        assert hint_indices == {x.index, y.index}

    def test_non_strict_returns_false_without_solution(self):
        """strict=False: no solution -> leave hints alone, return False."""
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        y = model.new_bool_var("y")
        model.add(x + y == 1)
        model.add(x + y == 2)  # infeasible
        # Pre-existing hint that must NOT be cleared on no-op.
        model.add_hint(x, 1)

        solver = cp_model.CpSolver()
        solver.solve(model)

        assert hint_from_solution(model, solver, strict=False) is False
        # Existing hint preserved.
        assert list(model.proto.solution_hint.vars) == [x.index]
        assert list(model.proto.solution_hint.values) == [1]

    def test_non_strict_returns_false_when_solve_not_called(self):
        model = cp_model.CpModel()
        model.new_bool_var("x")
        solver = cp_model.CpSolver()
        assert hint_from_solution(model, solver, strict=False) is False

    def test_returns_true_on_success(self):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        model.add(x == 1)
        solver = cp_model.CpSolver()
        solver.solve(model)
        assert hint_from_solution(model, solver) is True

    def test_round_trip_hint_is_feasible(self):
        """solve -> hint_from_solution -> assert_hint_feasible should pass."""
        model = cp_model.CpModel()
        x = model.new_int_var(0, 10, "x")
        y = model.new_int_var(0, 10, "y")
        model.add(x + y >= 5)
        model.minimize(x + 2 * y)

        solver = cp_model.CpSolver()
        solver.solve(model)
        hint_from_solution(model, solver)

        assert_hint_feasible(model)
