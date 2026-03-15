"""
Piecewise linear functions and constraints for CP-SAT.

CP-SAT operates on integers only, so piecewise linear functions generally
cannot be enforced as equalities (the true value may fall between integers).
Instead, this module provides upper-bound (y <= f(x)) and lower-bound
(y >= f(x)) constraints, plus floor/ceil/round equality constraints.
The implementation automatically partitions non-convex functions into convex
pieces and optionally adds a convex envelope as a redundant constraint.

Usage:
    from cpsat_utils.piecewise import PiecewiseLinearFunction
    import math

    model = cp_model.CpModel()
    x = model.new_int_var(0, 100, "x")

    # Approximate a non-linear function:
    f = PiecewiseLinearFunction.from_function(
        math.sqrt, x_min=0, x_max=100, num_breakpoints=20
    )
    y = f.add_round(model, x)   # y = round(sqrt(x))

    # One-sided bounds:
    g = PiecewiseLinearFunction([0, 10, 20], [0, 50, 30])
    y = g.add_upper_bound(model, x)  # y <= g(x)
    y = g.add_lower_bound(model, x)  # y >= g(x)

When to modify:
    - If CP-SAT adds native piecewise linear support
    - To add new bound types or formulation strategies
"""

from __future__ import annotations

import bisect
import fractions
import typing

from ortools.sat.python import cp_model

from cpsat_utils.piecewise._constraints import (
    _add_equality_constraint,
    _add_pwl_constraint,
)
from cpsat_utils.piecewise._helpers import _validate_bound_type


class PiecewiseLinearFunction:
    """
    A piecewise linear function defined by breakpoints (xs) and values (ys).

    The function interpolates linearly between consecutive points
    (xs[i], ys[i]) and (xs[i+1], ys[i+1]).

    Both xs and ys must be integers (CP-SAT requirement). The xs must be
    strictly increasing.

    Example::

        f = PiecewiseLinearFunction([0, 10, 20], [0, 50, 30])
        f(5)   # 25.0
        f(15)  # 40.0

        # Add as upper bound to a model:
        y = f.add_upper_bound(model, x)
    """

    __slots__ = ("xs", "ys")

    def __init__(self, xs: list[int], ys: list[int]) -> None:
        if len(xs) != len(ys):
            msg = f"xs and ys must have equal length, got {len(xs)} and {len(ys)}"
            raise ValueError(msg)
        if len(xs) < 2:
            msg = "Need at least 2 breakpoints"
            raise ValueError(msg)
        if any(x1 >= x2 for x1, x2 in zip(xs, xs[1:], strict=False)):
            msg = "xs must be strictly increasing"
            raise ValueError(msg)
        self.xs: list[int] = list(xs)
        self.ys: list[int] = list(ys)

    @classmethod
    def from_points(
        cls, points: typing.Iterable[tuple[int, int]]
    ) -> PiecewiseLinearFunction:
        """
        Create from an iterable of (x, y) pairs.

        Example::

            f = PiecewiseLinearFunction.from_points([(0, 0), (10, 50), (20, 30)])
        """
        pts = list(points)
        return cls([x for x, _ in pts], [y for _, y in pts])

    @classmethod
    def from_function(
        cls,
        f: typing.Callable[[int], int | float],
        *,
        xs: list[int] | None = None,
        x_min: int | None = None,
        x_max: int | None = None,
        num_breakpoints: int | None = None,
    ) -> PiecewiseLinearFunction:
        """
        Create by evaluating a callable at breakpoints.

        Breakpoints can be given explicitly via *xs*, or generated as
        *num_breakpoints* evenly spaced points from *x_min* to *x_max*.
        Values are rounded to the nearest integer.

        Examples::

            f = PiecewiseLinearFunction.from_function(lambda x: x**2, xs=[0, 5, 10])
            f = PiecewiseLinearFunction.from_function(
                math.sqrt, x_min=0, x_max=100, num_breakpoints=20
            )
        """
        if xs is not None:
            if any(v is not None for v in (x_min, x_max, num_breakpoints)):
                msg = "Provide either xs or (x_min, x_max, num_breakpoints), not both"
                raise ValueError(msg)
        else:
            if x_min is None or x_max is None or num_breakpoints is None:
                msg = "Provide either xs or all of (x_min, x_max, num_breakpoints)"
                raise ValueError(msg)
            if num_breakpoints < 2:
                msg = "num_breakpoints must be at least 2"
                raise ValueError(msg)
            step = (x_max - x_min) / (num_breakpoints - 1)
            xs = [x_min + round(i * step) for i in range(num_breakpoints)]
            # Ensure exact endpoints
            xs[0] = x_min
            xs[-1] = x_max
        return cls(xs, [round(f(x)) for x in xs])

    def __call__(self, x: int | float) -> float:
        """Evaluate the function at x via linear interpolation."""
        if not self.is_defined_for(x):
            msg = f"x={x} is outside [{self.xs[0]}, {self.xs[-1]}]"
            raise ValueError(msg)
        if x == self.xs[-1]:
            return float(self.ys[-1])
        i = bisect.bisect_right(self.xs, x) - 1
        dx = self.xs[i + 1] - self.xs[i]
        return self.ys[i] + (self.ys[i + 1] - self.ys[i]) * (x - self.xs[i]) / dx

    def is_defined_for(self, x: int | float) -> bool:
        """Check whether x is within the function's domain."""
        return self.xs[0] <= x <= self.xs[-1]

    @property
    def x_min(self) -> int:
        return self.xs[0]

    @property
    def x_max(self) -> int:
        return self.xs[-1]

    @property
    def y_min(self) -> int:
        return min(self.ys)

    @property
    def y_max(self) -> int:
        return max(self.ys)

    @property
    def num_segments(self) -> int:
        return len(self.xs) - 1

    def segment_gradients(self) -> list[fractions.Fraction]:
        """Return the exact gradient of each line segment as a Fraction."""
        return [
            fractions.Fraction(self.ys[i + 1] - self.ys[i], self.xs[i + 1] - self.xs[i])
            for i in range(self.num_segments)
        ]

    def segments(
        self,
    ) -> typing.Iterator[tuple[tuple[int, int], tuple[int, int]]]:
        """Yield ((x1, y1), (x2, y2)) for each segment."""
        for i in range(self.num_segments):
            yield (self.xs[i], self.ys[i]), (self.xs[i + 1], self.ys[i + 1])

    def is_convex(self, bound_type: str = "upper") -> bool:
        """
        Check whether the function is convex for the given bound type.

        For ``bound_type="upper"`` (y <= f(x)): the feasible region below
        the curve is convex when gradients are non-increasing.
        For ``bound_type="lower"`` (y >= f(x)): the feasible region above
        the curve is convex when gradients are non-decreasing.

        Uses integer cross-product comparison (no floating-point arithmetic).
        """
        _validate_bound_type(bound_type)
        for i in range(self.num_segments - 1):
            # Compare grad_i = dy_i/dx_i  vs  grad_{i+1} = dy_{i+1}/dx_{i+1}
            # using cross product: dy_i * dx_{i+1}  vs  dy_{i+1} * dx_i
            dy1 = self.ys[i + 1] - self.ys[i]
            dx1 = self.xs[i + 1] - self.xs[i]
            dy2 = self.ys[i + 2] - self.ys[i + 1]
            dx2 = self.xs[i + 2] - self.xs[i + 1]
            if bound_type == "upper":
                if dy1 * dx2 < dy2 * dx1:  # grad1 < grad2
                    return False
            else:
                if dy1 * dx2 > dy2 * dx1:  # grad1 > grad2
                    return False
        return True

    def copy(self) -> PiecewiseLinearFunction:
        return PiecewiseLinearFunction(list(self.xs), list(self.ys))

    def __repr__(self) -> str:
        return f"PiecewiseLinearFunction(xs={self.xs}, ys={self.ys})"

    # ------------------------------------------------------------------
    # Constraint construction
    # ------------------------------------------------------------------

    def add_upper_bound(
        self,
        model: cp_model.CpModel,
        x: cp_model.IntVar,
        *,
        y: cp_model.IntVar | None = None,
        name: str = "y",
        add_convex_envelope: bool = True,
        optimize_partition: bool = True,
    ) -> cp_model.IntVar:
        """
        Add ``y <= f(x)`` constraints to *model*.

        Args:
            model: The CP-SAT model.
            x: The input integer variable.
            y: Optional output variable. Created automatically if omitted.
            name: Name prefix for created variables (default ``"y"``).
                Use distinct names when adding multiple piecewise
                constraints to the same model.
            add_convex_envelope: Add redundant convex hull constraints
                (helps the solver; requires scipy for non-convex functions,
                silently skipped if unavailable).
            optimize_partition: Use minimal convex partition (recommended).

        Returns:
            The output variable *y* (created or passed through).
        """
        result = _add_pwl_constraint(
            model,
            x,
            self,
            bound_type="upper",
            y=y,
            name=name,
            add_convex_envelope=add_convex_envelope,
            optimize_partition=optimize_partition,
        )
        return result.y

    def add_lower_bound(
        self,
        model: cp_model.CpModel,
        x: cp_model.IntVar,
        *,
        y: cp_model.IntVar | None = None,
        name: str = "y",
        add_convex_envelope: bool = True,
        optimize_partition: bool = True,
    ) -> cp_model.IntVar:
        """
        Add ``y >= f(x)`` constraints to *model*.

        Args:
            model: The CP-SAT model.
            x: The input integer variable.
            y: Optional output variable. Created automatically if omitted.
            name: Name prefix for created variables (default ``"y"``).
            add_convex_envelope: Add redundant convex hull constraints.
            optimize_partition: Use minimal convex partition (recommended).

        Returns:
            The output variable *y* (created or passed through).
        """
        result = _add_pwl_constraint(
            model,
            x,
            self,
            bound_type="lower",
            y=y,
            name=name,
            add_convex_envelope=add_convex_envelope,
            optimize_partition=optimize_partition,
        )
        return result.y

    def add_floor(
        self,
        model: cp_model.CpModel,
        x: cp_model.IntVar,
        *,
        y: cp_model.IntVar | None = None,
        name: str = "y",
        add_convex_envelope: bool = True,
    ) -> cp_model.IntVar:
        """
        Add ``y = floor(f(x))`` constraints to *model*.

        Constrains y to be the largest integer not exceeding f(x).
        Uses per-segment selectors with both-sided constraints.

        Args:
            model: The CP-SAT model.
            x: The input integer variable.
            y: Optional output variable. Created automatically if omitted.
            name: Name prefix for created variables (default ``"y"``).
            add_convex_envelope: Add redundant convex hull constraints
                to help the solver bound y globally.

        Returns:
            The output variable *y*.
        """
        return _add_equality_constraint(
            model,
            x,
            self,
            "floor",
            y=y,
            name=name,
            add_convex_envelope=add_convex_envelope,
        )

    def add_ceil(
        self,
        model: cp_model.CpModel,
        x: cp_model.IntVar,
        *,
        y: cp_model.IntVar | None = None,
        name: str = "y",
        add_convex_envelope: bool = True,
    ) -> cp_model.IntVar:
        """
        Add ``y = ceil(f(x))`` constraints to *model*.

        Constrains y to be the smallest integer not less than f(x).
        Uses per-segment selectors with both-sided constraints.

        Args:
            model: The CP-SAT model.
            x: The input integer variable.
            y: Optional output variable. Created automatically if omitted.
            name: Name prefix for created variables (default ``"y"``).
            add_convex_envelope: Add redundant convex hull constraints
                to help the solver bound y globally.

        Returns:
            The output variable *y*.
        """
        return _add_equality_constraint(
            model,
            x,
            self,
            "ceil",
            y=y,
            name=name,
            add_convex_envelope=add_convex_envelope,
        )

    def add_round(
        self,
        model: cp_model.CpModel,
        x: cp_model.IntVar,
        *,
        y: cp_model.IntVar | None = None,
        name: str = "y",
        add_convex_envelope: bool = True,
    ) -> cp_model.IntVar:
        """
        Add ``y = round(f(x))`` constraints to *model*.

        Constrains y to be the nearest integer to f(x).
        Uses per-segment selectors with tightened both-sided constraints.

        Args:
            model: The CP-SAT model.
            x: The input integer variable.
            y: Optional output variable. Created automatically if omitted.
            name: Name prefix for created variables (default ``"y"``).
            add_convex_envelope: Add redundant convex hull constraints
                to help the solver bound y globally.

        Returns:
            The output variable *y*.
        """
        return _add_equality_constraint(
            model,
            x,
            self,
            "round",
            y=y,
            name=name,
            add_convex_envelope=add_convex_envelope,
        )
