"""
Tests for cpsat_utils context managers and assertion helpers.

Covers all public API: context managers (AssertModelFeasible,
AssertModelInfeasible, AssertObjectiveValue, AssertOptimalWithinTime)
and standalone functions (assert_feasible, assert_infeasible,
assert_optimal, assert_objective, solve).
"""

import pytest
from ortools.sat.python import cp_model

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

# --- Context Managers ---


class TestAssertModelFeasible:
    def test_feasible_model_passes(self):
        with AssertModelFeasible() as model:
            x = model.new_bool_var("x")
            y = model.new_bool_var("y")
            model.add(x + y == 1)

    def test_infeasible_model_raises(self):
        with (
            pytest.raises(RuntimeError, match="Expected feasible"),
            AssertModelFeasible() as model,
        ):
            x = model.new_bool_var("x")
            y = model.new_bool_var("y")
            model.add(x + y == 3)

    def test_accepts_existing_model(self):
        m = cp_model.CpModel()
        x = m.new_bool_var("x")
        m.add(x == 1)
        with AssertModelFeasible(model=m) as model:
            assert model is m

    def test_propagates_exception_from_body(self):
        with pytest.raises(ValueError, match="inner"), AssertModelFeasible():
            raise ValueError("inner")


class TestAssertModelInfeasible:
    def test_infeasible_model_passes(self):
        with AssertModelInfeasible() as model:
            x = model.new_bool_var("x")
            y = model.new_bool_var("y")
            model.add(x + y == 3)

    def test_feasible_model_raises(self):
        with (
            pytest.raises(RuntimeError, match="Expected infeasible"),
            AssertModelInfeasible() as model,
        ):
            x = model.new_bool_var("x")
            model.add(x == 1)


class TestAssertObjectiveValue:
    def test_correct_objective_passes(self):
        with AssertObjectiveValue(objective=1.0) as model:
            x = model.new_bool_var("x")
            y = model.new_bool_var("y")
            model.add(x + y >= 1)
            model.minimize(x + y)

    def test_wrong_objective_raises(self):
        with (
            pytest.raises(RuntimeError, match="Objective .* differs"),
            AssertObjectiveValue(objective=5.0) as model,
        ):
            x = model.new_bool_var("x")
            model.add(x == 1)
            model.minimize(x)

    def test_infeasible_model_raises(self):
        with (
            pytest.raises(RuntimeError, match="Expected feasible"),
            AssertObjectiveValue(objective=0.0) as model,
        ):
            x = model.new_bool_var("x")
            model.add(x >= 2)
            model.minimize(x)

    def test_maximize_objective(self):
        with AssertObjectiveValue(objective=2.0) as model:
            x = model.new_bool_var("x")
            y = model.new_bool_var("y")
            model.maximize(x + y)


class TestAssertOptimalWithinTime:
    def test_optimal_passes(self):
        with AssertOptimalWithinTime(time_limit=5.0) as model:
            x = model.new_bool_var("x")
            model.add(x == 1)
            model.minimize(x)

    def test_infeasible_raises(self):
        with (
            pytest.raises(RuntimeError, match="Expected optimal"),
            AssertOptimalWithinTime() as model,
        ):
            x = model.new_bool_var("x")
            model.add(x >= 2)
            model.minimize(x)


# --- Standalone Functions ---


class TestAssertFeasible:
    def test_feasible(self):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        model.add(x == 1)
        solver = assert_feasible(model)
        assert solver.value(x) == 1

    def test_infeasible_raises(self):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        model.add(x >= 2)
        with pytest.raises(AssertionError, match="Expected feasible"):
            assert_feasible(model)


class TestAssertInfeasible:
    def test_infeasible(self):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        model.add(x >= 2)
        assert_infeasible(model)

    def test_feasible_raises(self):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        model.add(x == 1)
        with pytest.raises(AssertionError, match="Expected infeasible"):
            assert_infeasible(model)


class TestAssertOptimal:
    def test_optimal(self):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        model.add(x == 1)
        model.minimize(x)
        assert_optimal(model)

    def test_infeasible_raises(self):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        model.add(x >= 2)
        with pytest.raises(AssertionError, match="Expected optimal"):
            assert_optimal(model)


class TestAssertObjective:
    def test_correct_objective(self):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        y = model.new_bool_var("y")
        model.add(x + y >= 1)
        model.minimize(x + y)
        solver = assert_objective(model, expected=1.0)
        assert solver is not None

    def test_wrong_objective_raises(self):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        model.add(x == 1)
        model.minimize(x)
        with pytest.raises(AssertionError, match="Expected objective"):
            assert_objective(model, expected=5.0)

    def test_infeasible_raises(self):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        model.add(x >= 2)
        model.minimize(x)
        with pytest.raises(AssertionError, match="Expected feasible"):
            assert_objective(model, expected=0.0)

    def test_with_explicit_solver(self):
        """Matches the usage pattern from the TDD chapter."""
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        model.add(x == 1)
        model.minimize(x)
        solver = cp_model.CpSolver()
        assert_objective(model=model, solver=solver, expected=1.0)
        assert solver.value(x) == 1


class TestSolve:
    def test_solve_optimal(self):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        model.add(x == 1)
        model.minimize(x)
        solver = solve(model)
        assert solver.value(x) == 1

    def test_solve_expect_list(self):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        model.add(x == 1)
        model.minimize(x)
        solver = solve(model, expect=[cp_model.OPTIMAL, cp_model.FEASIBLE])
        assert solver.value(x) == 1

    def test_solve_wrong_status_raises(self):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        model.add(x >= 2)
        model.minimize(x)
        with pytest.raises(AssertionError, match="Expected status"):
            solve(model)

    def test_solve_with_time_limit(self):
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        model.add(x == 1)
        model.minimize(x)
        solver = solve(model, time_limit=5.0)
        assert solver.value(x) == 1
