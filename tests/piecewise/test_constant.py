"""Tests for step (piecewise constant) functions and constraints."""

import pytest
from ortools.sat.python import cp_model

from cpsat_utils.piecewise import StepFunction
from cpsat_utils.testing import assert_objective

# ---------------------------------------------------------------------------
# StepFunction
# ---------------------------------------------------------------------------


class TestStepFunction:
    def test_evaluation(self):
        f = StepFunction([0, 3, 7, 10], [10, 20, 30])
        assert f(0) == 10
        assert f(2) == 10
        assert f(3) == 20
        assert f(6) == 20
        assert f(7) == 30
        assert f(9) == 30

    def test_last_boundary_inclusive(self):
        """The last boundary is inclusive, unlike intermediate boundaries."""
        f = StepFunction([0, 5, 10], [1, 2])
        assert f(10) == 2  # last piece includes right endpoint

    def test_out_of_bounds(self):
        f = StepFunction([0, 5, 10], [1, 2])
        with pytest.raises(ValueError):
            f(-1)
        with pytest.raises(ValueError):
            f(11)

    def test_validation_length(self):
        with pytest.raises(ValueError, match="len.xs. must be len.ys. \\+ 1"):
            StepFunction([0, 10], [1, 2])

    def test_validation_increasing(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            StepFunction([0, 0, 10], [1, 2])

    def test_monotone(self):
        assert StepFunction([0, 1, 2, 3], [0, 1, 2]).is_monotone()
        assert StepFunction([0, 1, 2, 3], [2, 1, 0]).is_monotone()
        assert not StepFunction([0, 1, 2, 3], [0, 2, 1]).is_monotone()

    def test_monotone_single_piece(self):
        """Single piece is trivially monotone."""
        assert StepFunction([0, 10], [5]).is_monotone()

    def test_monotone_all_equal(self):
        """Constant function is monotone (both non-decreasing and non-increasing)."""
        assert StepFunction([0, 1, 2, 3], [5, 5, 5]).is_monotone()

    def test_validation_zero_pieces(self):
        with pytest.raises(ValueError, match="at least 1"):
            StepFunction([0], [])

    def test_negative_values(self):
        """Negative y values should work correctly."""
        f = StepFunction([0, 5, 10], [-10, -20])
        assert f(0) == -10
        assert f(7) == -20

    def test_domain(self):
        f = StepFunction([0, 5, 10], [1, 2])
        assert f.x_min == 0
        assert f.x_max == 10
        assert f.is_defined_for(0)
        assert f.is_defined_for(10)
        assert not f.is_defined_for(11)


class TestSimplified:
    def test_merges_equal_adjacent(self):
        f = StepFunction([0, 3, 7, 10], [10, 10, 30])
        g = f.simplified()
        assert g.xs == [0, 7, 10]
        assert g.ys == [10, 30]

    def test_all_equal(self):
        """All pieces have the same value — collapses to single piece."""
        f = StepFunction([0, 3, 7, 10], [5, 5, 5])
        g = f.simplified()
        assert g.xs == [0, 10]
        assert g.ys == [5]

    def test_no_merge_needed(self):
        """All values distinct — no change."""
        f = StepFunction([0, 3, 7, 10], [10, 20, 30])
        g = f.simplified()
        assert g.xs == f.xs
        assert g.ys == f.ys

    def test_single_piece(self):
        f = StepFunction([0, 10], [42])
        g = f.simplified()
        assert g.xs == [0, 10]
        assert g.ys == [42]

    def test_preserves_function_values(self):
        """Simplified function returns same values for all x (inclusive domain)."""
        f = StepFunction([0, 2, 5, 8, 10], [10, 10, 30, 30])
        g = f.simplified()
        for x in range(11):  # 0..10 inclusive
            assert f(x) == g(x), f"Mismatch at x={x}"


class TestFromIntervals:
    def test_basic(self):
        f = StepFunction.from_intervals([(0, 10), (3, 20), (7, 30)], x_max=10)
        assert f.xs == [0, 3, 7, 10]
        assert f.ys == [10, 20, 30]
        assert f(0) == 10
        assert f(5) == 20
        assert f(10) == 30  # inclusive

    def test_unsorted_input(self):
        """Intervals are sorted by start."""
        f = StepFunction.from_intervals([(7, 30), (0, 10), (3, 20)], x_max=10)
        assert f.xs == [0, 3, 7, 10]
        assert f.ys == [10, 20, 30]

    def test_single_interval(self):
        """Single interval produces a one-piece function."""
        f = StepFunction.from_intervals([(0, 42)], x_max=10)
        assert f.xs == [0, 10]
        assert f.ys == [42]
        assert f(5) == 42


# ---------------------------------------------------------------------------
# add_constraint
# ---------------------------------------------------------------------------


def _make_pcf_model(
    f: StepFunction, **kwargs
) -> tuple[cp_model.CpModel, cp_model.IntVar, cp_model.IntVar]:
    """Helper: create model + x + y with a step-function constraint."""
    model = cp_model.CpModel()
    x = model.new_int_var(0, f.xs[-1], "x")
    y = f.add_constraint(model, x, **kwargs)
    return model, x, y


class TestAddConstraint:
    def test_maximize_stairs(self):
        f = StepFunction([0, 1, 2, 3], [0, 1, 2])
        model, x, y = _make_pcf_model(f)
        model.maximize(y)
        solver = assert_objective(model, 2)
        assert solver.value(x) in (2, 3)  # last piece is [2, 3] inclusive

    def test_minimize_stairs(self):
        f = StepFunction([0, 1, 2, 3], [0, 1, 2])
        model, x, y = _make_pcf_model(f)
        model.minimize(y)
        solver = assert_objective(model, 0)
        assert solver.value(x) == 0

    def test_pyramid(self):
        f = StepFunction([0, 1, 2, 3], [0, 1, 0])
        model, x, y = _make_pcf_model(f)
        model.maximize(y)
        solver = assert_objective(model, 1)
        assert solver.value(x) == 1

    def test_larger_pyramid(self):
        f = StepFunction([0, 1, 2, 3, 4, 5], [0, 1, 5, 1, 0])
        model, x, y = _make_pcf_model(f)
        model.maximize(y)
        solver = assert_objective(model, 5)
        assert solver.value(x) in (2, 3)  # last piece includes right endpoint

    def test_wide_intervals(self):
        """Each piece spans multiple x values."""
        f = StepFunction([0, 20, 50, 100], [10, 30, 5])
        model, x, y = _make_pcf_model(f)
        model.maximize(y)
        solver = assert_objective(model, 30)
        assert 20 <= solver.value(x) < 50

    def test_restrict_domain(self):
        f = StepFunction([0, 3, 7, 10], [10, 30, 50])
        model, x, y = _make_pcf_model(f, restrict_domain=True)
        model.minimize(y)
        assert_objective(model, 10)

    def test_single_piece(self):
        """Edge case: only one piece, no step variables needed."""
        f = StepFunction([0, 10], [42])
        model, x, y = _make_pcf_model(f)
        model.maximize(y)
        assert_objective(model, 42)

    def test_user_provided_y(self):
        f = StepFunction([0, 1, 2, 3], [10, 20, 30])
        model = cp_model.CpModel()
        x = model.new_int_var(0, 10, "x")
        y = model.new_int_var(0, 100, "my_y")
        returned_y = f.add_constraint(model, x, y=y)
        assert returned_y is y
        model.maximize(y)
        assert_objective(model, 30)

    def test_negative_values(self):
        """Constraint with negative y values."""
        f = StepFunction([0, 5, 10], [-10, -20])
        model, x, y = _make_pcf_model(f)
        model.maximize(y)
        assert_objective(model, -10)

    def test_all_equal_values(self):
        """All pieces have the same value."""
        f = StepFunction([0, 3, 6, 9], [7, 7, 7])
        model, x, y = _make_pcf_model(f)
        model.maximize(y)
        assert_objective(model, 7)

    def test_two_pieces(self):
        """Two-piece function: one step variable."""
        f = StepFunction([0, 5, 10], [10, 30])
        model, x, y = _make_pcf_model(f)
        model.maximize(y)
        solver = assert_objective(model, 30)
        assert 5 <= solver.value(x) <= 10  # inclusive

    def test_last_piece_includes_boundary(self):
        """The constraint must allow x == xs[-1]."""
        f = StepFunction([0, 5, 10], [10, 30])
        model = cp_model.CpModel()
        x = model.new_int_var(10, 10, "x")  # force x to right boundary
        y = f.add_constraint(model, x, name="v")
        model.maximize(y)
        assert_objective(model, 30)

    def test_from_intervals_integration(self):
        """Constructor from_intervals works with add_constraint."""
        f = StepFunction.from_intervals([(0, 10), (5, 50), (8, 20)], x_max=10)
        model, x, y = _make_pcf_model(f)
        model.maximize(y)
        assert_objective(model, 50)
