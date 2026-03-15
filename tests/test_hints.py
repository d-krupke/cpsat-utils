"""
Tests for cpsat_utils.hints (assert_hint_feasible, complete_hint).
"""

import pytest
from ortools.sat.python import cp_model

from cpsat_utils.hints import assert_hint_feasible, complete_hint


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
