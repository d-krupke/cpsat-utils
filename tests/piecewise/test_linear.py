"""Tests for piecewise linear functions and constraints."""

import fractions
import math

import pytest
from ortools.sat.python import cp_model

from cpsat_utils.piecewise import PiecewiseLinearFunction
from cpsat_utils.piecewise._helpers import (
    _are_collinear,
    _convex_envelope,
    _integer_line_coefficients,
    _simplify,
    _split_into_convex_parts,
)
from cpsat_utils.testing import assert_objective

# ---------------------------------------------------------------------------
# PiecewiseLinearFunction
# ---------------------------------------------------------------------------


class TestPiecewiseLinearFunction:
    def test_evaluation(self):
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 5])
        assert f(0) == 0
        assert f(5) == 5
        assert f(10) == 10
        assert round(f(16)) == 7
        assert f(20) == 5

    def test_out_of_bounds(self):
        f = PiecewiseLinearFunction([0, 10], [0, 10])
        with pytest.raises(ValueError):
            f(-1)
        with pytest.raises(ValueError):
            f(11)

    def test_validation_length_mismatch(self):
        with pytest.raises(ValueError, match="equal length"):
            PiecewiseLinearFunction([0, 10], [0])

    def test_validation_not_increasing(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            PiecewiseLinearFunction([0, 0], [0, 10])

    def test_validation_too_few_points(self):
        with pytest.raises(ValueError, match="at least 2"):
            PiecewiseLinearFunction([0], [0])

    def test_convex_upper(self):
        """Concave-shaped function (gradients decrease) is convex for upper bound."""
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 5])
        assert f.is_convex("upper")
        assert not f.is_convex("lower")

    def test_convex_lower(self):
        """Convex-shaped function (gradients increase) is convex for lower bound."""
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 50])
        assert not f.is_convex("upper")
        assert f.is_convex("lower")

    def test_properties(self):
        f = PiecewiseLinearFunction([0, 10, 20], [5, 15, 10])
        assert f.x_min == 0
        assert f.x_max == 20
        assert f.y_min == 5
        assert f.y_max == 15
        assert f.num_segments == 2

    def test_segments(self):
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 5])
        segs = list(f.segments())
        assert segs == [((0, 0), (10, 10)), ((10, 10), (20, 5))]


# ---------------------------------------------------------------------------
# Alternative constructors
# ---------------------------------------------------------------------------


class TestConstructors:
    def test_from_points(self):
        f = PiecewiseLinearFunction.from_points([(0, 0), (10, 50), (20, 30)])
        assert f.xs == [0, 10, 20]
        assert f.ys == [0, 50, 30]
        assert f(10) == 50

    def test_from_function(self):
        f = PiecewiseLinearFunction.from_function(lambda x: x * x, xs=[0, 5, 10])
        assert f.xs == [0, 5, 10]
        assert f.ys == [0, 25, 100]

    def test_from_function_rounds(self):
        f = PiecewiseLinearFunction.from_function(lambda x: x / 3, xs=[0, 10, 20])
        assert f.ys == [0, 3, 7]  # rounded

    def test_from_function_with_range(self):
        f = PiecewiseLinearFunction.from_function(
            lambda x: x * x, x_min=0, x_max=10, num_breakpoints=6
        )
        assert f.xs[0] == 0
        assert f.xs[-1] == 10
        assert len(f.xs) == 6
        assert f.ys[0] == 0
        assert f.ys[-1] == 100

    def test_from_function_rejects_both(self):
        with pytest.raises(ValueError, match="not both"):
            PiecewiseLinearFunction.from_function(
                lambda x: x, xs=[0, 10], x_min=0, x_max=10, num_breakpoints=5
            )

    def test_from_function_requires_all_range_params(self):
        with pytest.raises(ValueError, match="all of"):
            PiecewiseLinearFunction.from_function(lambda x: x, x_min=0, x_max=10)

    def test_from_function_min_breakpoints(self):
        """num_breakpoints=2 should produce exactly 2 points."""
        f = PiecewiseLinearFunction.from_function(
            lambda x: x * 2, x_min=0, x_max=10, num_breakpoints=2
        )
        assert f.xs == [0, 10]
        assert f.ys == [0, 20]

    def test_from_function_rejects_too_few_breakpoints(self):
        with pytest.raises(ValueError, match="at least 2"):
            PiecewiseLinearFunction.from_function(
                lambda x: x, x_min=0, x_max=10, num_breakpoints=1
            )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_coordinates(self):
        """Functions with negative x and y values should work."""
        f = PiecewiseLinearFunction([-10, 0, 10], [-5, 10, -3])
        assert f(-10) == -5
        assert f(0) == 10
        assert f(10) == -3

    def test_single_segment_is_convex(self):
        """A single segment is convex for both bound types."""
        f = PiecewiseLinearFunction([0, 10], [0, 10])
        assert f.is_convex("upper")
        assert f.is_convex("lower")

    def test_flat_function_is_convex(self):
        """A constant function is convex for both bound types."""
        f = PiecewiseLinearFunction([0, 10, 20], [5, 5, 5])
        assert f.is_convex("upper")
        assert f.is_convex("lower")

    def test_is_convex_invalid_bound_type(self):
        f = PiecewiseLinearFunction([0, 10], [0, 10])
        with pytest.raises(ValueError, match="bound_type"):
            f.is_convex("invalid")


class TestHelpers:
    def test_collinear(self):
        assert _are_collinear((0, 0), (10, 10), (20, 20))
        assert _are_collinear((0, 1), (10, 11), (20, 21))
        assert not _are_collinear((0, 0), (10, 10), (20, 21))

    def test_simplify(self):
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 20])
        s = _simplify(f)
        assert len(s.xs) == 2  # middle point is redundant

    def test_simplify_no_change(self):
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 5])
        s = _simplify(f)
        assert len(s.xs) == 3

    def test_integer_line_coefficients(self):
        assert _integer_line_coefficients(0, 0, 10, 10) == (1, 1, 0)
        assert _integer_line_coefficients(0, 0, 20, 10) == (2, 1, 0)
        assert _integer_line_coefficients(0, 0, 10, 15) == (2, 3, 0)
        assert _integer_line_coefficients(0, 0, 10, -10) == (1, -1, 0)

    def test_integer_line_flat(self):
        t, a, b = _integer_line_coefficients(0, 5, 10, 5)
        assert a == 0
        assert b == 5 * t

    def test_integer_line_negative_coords(self):
        """Line through negative coordinates produces valid coefficients."""
        t, a, b = _integer_line_coefficients(-10, -5, 10, 15)
        # Verify: t*y == a*x + b at both endpoints
        assert t * (-5) == a * (-10) + b
        assert t * 15 == a * 10 + b
        assert t > 0

    def test_integer_line_steep_slope(self):
        """Steep slope: dy=100, dx=1."""
        t, a, b = _integer_line_coefficients(0, 0, 1, 100)
        assert t * 0 == a * 0 + b
        assert t * 100 == a * 1 + b
        assert t > 0

    def test_split_convex_upper(self):
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 50])
        parts = _split_into_convex_parts(f, "upper")
        assert len(parts) == 2
        assert all(p.is_convex("upper") for p in parts)

    def test_split_already_convex(self):
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 5])
        parts = _split_into_convex_parts(f, "upper")
        assert len(parts) == 1

    def test_split_convex_lower(self):
        """Non-convex for lower bound needs splitting."""
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 5])
        parts = _split_into_convex_parts(f, "lower")
        assert len(parts) == 2
        assert all(p.is_convex("lower") for p in parts)

    def test_split_many_segments(self):
        """Zigzag function: each direction change creates a new part."""
        f = PiecewiseLinearFunction([0, 10, 20, 30, 40], [0, 10, 0, 10, 0])
        parts = _split_into_convex_parts(f, "upper")
        assert len(parts) >= 2
        assert all(p.is_convex("upper") for p in parts)


class TestConvexEnvelope:
    def test_upper_envelope(self):
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 50])
        g = _convex_envelope(f, "upper")
        assert g.is_convex("upper")
        # Envelope must be >= original at all integer points
        for x in range(21):
            assert g(x) >= f(x) - 1e-9

    def test_lower_envelope(self):
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 50])
        g = _convex_envelope(f, "lower")
        assert g.is_convex("lower")
        for x in range(21):
            assert g(x) <= f(x) + 1e-9

    def test_already_convex(self):
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 5])
        g = _convex_envelope(f, "upper")
        assert len(g.xs) == len(f.xs)

    def test_envelope_many_breakpoints(self):
        """Envelope of a zigzag should have fewer points than the original."""
        xs = list(range(0, 51, 10))
        ys = [0, 30, 10, 40, 20, 50]
        f = PiecewiseLinearFunction(xs, ys)
        g = _convex_envelope(f, "upper")
        assert g.is_convex("upper")
        for x in range(51):
            assert g(x) >= f(x) - 1e-9


# ---------------------------------------------------------------------------
# add_upper_bound / add_lower_bound
# ---------------------------------------------------------------------------


def _make_model_with_bound(
    f: PiecewiseLinearFunction, bound: str, **kwargs
) -> tuple[cp_model.CpModel, cp_model.IntVar, cp_model.IntVar]:
    """Helper: create model + x + y with the given bound constraint."""
    model = cp_model.CpModel()
    x = model.new_int_var(f.x_min, f.x_max, "x")
    method = getattr(f, f"add_{bound}")
    y = method(model, x, **kwargs)
    return model, x, y


class TestAddUpperBound:
    def test_convex_function(self):
        """Concave function: single convex part, no auxiliary variables."""
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 5])
        model, x, y = _make_model_with_bound(f, "upper_bound")
        model.maximize(y)
        solver = assert_objective(model, 10)
        assert solver.value(x) == 10

    def test_non_convex_function(self):
        """Non-convex function needs reified constraints."""
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 50])
        model, x, y = _make_model_with_bound(f, "upper_bound")
        model.maximize(y)
        solver = assert_objective(model, 50)
        assert solver.value(x) == 20

    def test_flat_segment(self):
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 10])
        model, x, y = _make_model_with_bound(f, "upper_bound")
        model.maximize(y)
        assert_objective(model, 10)

    def test_user_provided_y(self):
        model = cp_model.CpModel()
        x = model.new_int_var(0, 20, "x")
        y = model.new_int_var(-100, 100, "y")
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 5])
        returned_y = f.add_upper_bound(model, x, y=y)
        assert returned_y is y
        model.maximize(y)
        assert_objective(model, 10)

    def test_many_segments(self):
        """Stress test with many breakpoints."""
        xs = list(range(0, 101, 5))  # 21 points
        ys = [x * (100 - x) for x in xs]  # parabola-like
        f = PiecewiseLinearFunction(xs, ys)
        model, x, y = _make_model_with_bound(f, "upper_bound")
        model.maximize(y)
        solver = assert_objective(model, 2500)
        assert solver.value(x) == 50

    def test_from_points_integration(self):
        """Constructor from_points works with add_upper_bound."""
        f = PiecewiseLinearFunction.from_points([(0, 0), (10, 10), (20, 5)])
        model, x, y = _make_model_with_bound(f, "upper_bound")
        model.maximize(y)
        assert_objective(model, 10)

    def test_single_segment(self):
        """Two breakpoints: single segment, always convex."""
        f = PiecewiseLinearFunction([0, 10], [0, 50])
        model, x, y = _make_model_with_bound(f, "upper_bound")
        model.maximize(y)
        assert_objective(model, 50)

    def test_negative_values(self):
        """Function with negative x and y domain."""
        f = PiecewiseLinearFunction([-10, 0, 10], [10, -5, 10])
        model = cp_model.CpModel()
        x = model.new_int_var(-10, 10, "x")
        y = f.add_upper_bound(model, x)
        model.minimize(y)
        # y <= f(x), minimizing y pushes it to y_min=-5; any x is valid
        assert_objective(model, -5)

    def test_descending_function(self):
        """Monotonically decreasing function."""
        f = PiecewiseLinearFunction([0, 10, 20], [100, 50, 0])
        model, x, y = _make_model_with_bound(f, "upper_bound")
        model.maximize(y)
        assert_objective(model, 100)

    def test_zigzag(self):
        """Multiple direction changes: up-down-up-down."""
        f = PiecewiseLinearFunction([0, 10, 20, 30, 40], [0, 20, 5, 25, 10])
        model, x, y = _make_model_with_bound(f, "upper_bound")
        model.maximize(y)
        assert_objective(model, 25)


class TestAddLowerBound:
    def test_basic(self):
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 50])
        model, x, y = _make_model_with_bound(f, "lower_bound")
        model.minimize(y)
        solver = assert_objective(model, 0)
        assert solver.value(x) == 0

    def test_interior_min(self):
        f = PiecewiseLinearFunction([0, 10, 20, 30], [20, 10, 50, 40])
        model, x, y = _make_model_with_bound(f, "lower_bound")
        model.minimize(y)
        solver = assert_objective(model, 10)
        assert solver.value(x) == 10

    def test_single_segment(self):
        f = PiecewiseLinearFunction([0, 10], [0, 50])
        model, x, y = _make_model_with_bound(f, "lower_bound")
        model.minimize(y)
        assert_objective(model, 0)

    def test_negative_values(self):
        """Lower bound with negative values."""
        f = PiecewiseLinearFunction([-10, 0, 10], [-20, 10, -30])
        model = cp_model.CpModel()
        x = model.new_int_var(-10, 10, "x")
        y = f.add_lower_bound(model, x)
        model.minimize(y)
        solver = assert_objective(model, -30)
        assert solver.value(x) == 10

    def test_user_provided_y(self):
        model = cp_model.CpModel()
        x = model.new_int_var(0, 20, "x")
        y = model.new_int_var(-100, 100, "y")
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 50])
        returned_y = f.add_lower_bound(model, x, y=y)
        assert returned_y is y


# ---------------------------------------------------------------------------
# add_floor / add_ceil / add_round
# ---------------------------------------------------------------------------


def _assert_equality_mode(
    f: PiecewiseLinearFunction, mode: str, xi: int, expected: int
) -> None:
    """Helper: assert that add_{mode}(model, x) with x==xi gives expected."""
    model = cp_model.CpModel()
    x = model.new_int_var(f.x_min, f.x_max, "x")
    method = getattr(f, f"add_{mode}")
    y = method(model, x)
    model.add(x == xi)
    # Use maximize for floor/round, minimize for ceil to stress both directions
    if mode == "ceil":
        model.minimize(y)
    else:
        model.maximize(y)
    assert_objective(model, expected)


class TestAddFloor:
    def test_integer_valued(self):
        """When f(x) is integer at all integer x, floor == f(x)."""
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 20])
        _assert_equality_mode(f, "floor", 15, 15)

    def test_non_integer(self):
        """f(x) = x * 10/3, so f(1) = 3.33..., floor = 3."""
        f = PiecewiseLinearFunction([0, 3], [0, 10])
        _assert_equality_mode(f, "floor", 1, 3)

    def test_all_values_correct(self):
        """Check floor for all x in a non-trivial function."""
        f = PiecewiseLinearFunction([0, 3, 6], [0, 10, 4])
        for xi in range(7):
            _assert_equality_mode(f, "floor", xi, math.floor(f(xi)))

    def test_user_provided_y(self):
        f = PiecewiseLinearFunction([0, 10], [0, 10])
        model = cp_model.CpModel()
        x = model.new_int_var(0, 10, "x")
        y = model.new_int_var(-100, 100, "y")
        returned_y = f.add_floor(model, x, y=y)
        assert returned_y is y


class TestAddCeil:
    def test_integer_valued(self):
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 20])
        _assert_equality_mode(f, "ceil", 15, 15)

    def test_non_integer(self):
        """f(1) = 10/3 = 3.33..., ceil = 4."""
        f = PiecewiseLinearFunction([0, 3], [0, 10])
        _assert_equality_mode(f, "ceil", 1, 4)

    def test_all_values_correct(self):
        f = PiecewiseLinearFunction([0, 3, 6], [0, 10, 4])
        for xi in range(7):
            _assert_equality_mode(f, "ceil", xi, math.ceil(f(xi)))


class TestAddRound:
    def test_integer_valued(self):
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 20])
        _assert_equality_mode(f, "round", 15, 15)

    def test_non_integer(self):
        """f(1) = 10/3 = 3.33..., round = 3."""
        f = PiecewiseLinearFunction([0, 3], [0, 10])
        _assert_equality_mode(f, "round", 1, 3)

    def test_all_values_correct(self):
        f = PiecewiseLinearFunction([0, 3, 6], [0, 10, 4])
        for xi in range(7):
            _assert_equality_mode(f, "round", xi, round(f(xi)))

    def test_from_function_integration(self):
        """Full pipeline: from_function + add_round."""
        f = PiecewiseLinearFunction.from_function(
            lambda x: x**2, x_min=0, x_max=10, num_breakpoints=11
        )
        _assert_equality_mode(f, "round", 7, 49)


# ---------------------------------------------------------------------------
# Non-convex equality constraints
# ---------------------------------------------------------------------------


class TestEqualityNonConvex:
    """floor/ceil/round on functions requiring segment selectors."""

    def test_floor_non_convex(self):
        """Non-convex function: zigzag with fractional values."""
        f = PiecewiseLinearFunction([0, 3, 6, 9], [0, 10, 2, 12])
        for xi in range(10):
            _assert_equality_mode(f, "floor", xi, math.floor(f(xi)))

    def test_ceil_non_convex(self):
        f = PiecewiseLinearFunction([0, 3, 6, 9], [0, 10, 2, 12])
        for xi in range(10):
            _assert_equality_mode(f, "ceil", xi, math.ceil(f(xi)))

    def test_round_non_convex(self):
        f = PiecewiseLinearFunction([0, 3, 6, 9], [0, 10, 2, 12])
        for xi in range(10):
            _assert_equality_mode(f, "round", xi, round(f(xi)))

    def test_floor_single_segment(self):
        """Single segment: no selector variables needed."""
        f = PiecewiseLinearFunction([0, 3], [0, 10])
        for xi in range(4):
            _assert_equality_mode(f, "floor", xi, math.floor(f(xi)))

    def test_ceil_single_segment(self):
        f = PiecewiseLinearFunction([0, 3], [0, 10])
        for xi in range(4):
            _assert_equality_mode(f, "ceil", xi, math.ceil(f(xi)))

    def test_floor_negative_slope(self):
        """Descending function."""
        f = PiecewiseLinearFunction([0, 7], [10, 0])
        for xi in range(8):
            _assert_equality_mode(f, "floor", xi, math.floor(f(xi)))

    def test_ceil_negative_slope(self):
        f = PiecewiseLinearFunction([0, 7], [10, 0])
        for xi in range(8):
            _assert_equality_mode(f, "ceil", xi, math.ceil(f(xi)))

    def test_floor_negative_values(self):
        """Function dipping below zero."""
        f = PiecewiseLinearFunction([0, 5, 10], [5, -5, 5])
        for xi in range(11):
            _assert_equality_mode(f, "floor", xi, math.floor(f(xi)))

    def test_ceil_negative_values(self):
        f = PiecewiseLinearFunction([0, 5, 10], [5, -5, 5])
        for xi in range(11):
            _assert_equality_mode(f, "ceil", xi, math.ceil(f(xi)))

    def test_floor_flat_segment(self):
        """Flat segment: f(x) is always integer, floor == f(x)."""
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 10])
        for xi in range(21):
            _assert_equality_mode(f, "floor", xi, math.floor(f(xi)))

    def test_round_steep_slope(self):
        """Steep slope where rounding boundary is exercised heavily."""
        f = PiecewiseLinearFunction([0, 3], [0, 100])
        for xi in range(4):
            _assert_equality_mode(f, "round", xi, round(f(xi)))


# ---------------------------------------------------------------------------
# segment_gradients returns exact Fractions
# ---------------------------------------------------------------------------


class TestSegmentGradients:
    def test_returns_fractions(self):
        f = PiecewiseLinearFunction([0, 10, 20], [0, 10, 5])
        grads = f.segment_gradients()
        assert all(isinstance(g, fractions.Fraction) for g in grads)

    def test_exact_values(self):
        f = PiecewiseLinearFunction([0, 3, 6], [0, 10, 4])
        grads = f.segment_gradients()
        assert grads[0] == fractions.Fraction(10, 3)
        assert grads[1] == fractions.Fraction(-6, 3)

    def test_large_coordinates_exact(self):
        """Float division loses precision for large values; Fraction does not."""
        # 10**17 + 1 and 10**17 differ by 1, but float can't distinguish
        big = 10**17
        f = PiecewiseLinearFunction([0, big], [0, big + 1])
        grad = f.segment_gradients()[0]
        # Exact: (10^17 + 1) / 10^17, NOT equal to 1
        assert grad != 1
        assert grad == fractions.Fraction(big + 1, big)

    def test_comparison_with_fractions(self):
        """Gradient comparisons are exact when using Fraction."""
        f = PiecewiseLinearFunction([0, 3, 7], [0, 10, 23])
        grads = f.segment_gradients()
        # grad[0] = 10/3, grad[1] = 13/4
        # 10/3 ≈ 3.333, 13/4 = 3.25 → grad[0] > grad[1]
        assert grads[0] > grads[1]


# ---------------------------------------------------------------------------
# _integer_line_coefficients invariant checks survive python -O
# ---------------------------------------------------------------------------


class TestIntegerLineInvariants:
    def test_coefficients_satisfy_line_equation(self):
        """Verify t*y == a*x + b holds at both endpoints for various lines."""
        test_cases = [
            (0, 0, 10, 10),
            (-5, -3, 5, 7),
            (0, 0, 7, 3),
            (0, 0, 3, 7),
            (100, 200, 300, 400),
        ]
        for x0, y0, x1, y1 in test_cases:
            t, a, b = _integer_line_coefficients(x0, y0, x1, y1)
            assert t > 0, f"t must be positive for ({x0},{y0})->({x1},{y1})"
            assert t * y0 == a * x0 + b, f"Line eq fails at ({x0},{y0})"
            assert t * y1 == a * x1 + b, f"Line eq fails at ({x1},{y1})"
