"""
Step (piecewise constant) functions and constraints for CP-SAT.

A step function maps integer intervals to constant values:
f(x) = ys[i] for xs[i] <= x < xs[i+1], with the last piece covering
xs[-2] <= x <= xs[-1] (inclusive). Unlike piecewise linear functions,
equality constraints (y == f(x)) are well-defined since the function
values are integers.

Usage:
    from cpsat_utils.piecewise import StepFunction

    model = cp_model.CpModel()
    x = model.new_int_var(0, 10, "x")
    f = StepFunction([0, 3, 7, 10], [10, 20, 30])
    y = f.add_constraint(model, x)

    # From (interval_start, value) pairs:
    f = StepFunction.from_intervals(
        [(0, 10), (3, 20), (7, 30)], x_max=10
    )

When to modify:
    - If CP-SAT adds native step-function support
    - To add alternative encodings for specific function shapes
"""

from __future__ import annotations

import bisect
import typing

from ortools.sat.python import cp_model


class StepFunction:
    """
    A step function defined by boundaries (xs) and interval values (ys).

    The function has ``len(ys)`` constant pieces. Piece *i* covers
    ``xs[i] <= x < xs[i+1]``, except the last piece which includes
    its right endpoint: ``xs[-2] <= x <= xs[-1]``. This makes the
    domain ``[xs[0], xs[-1]]`` (inclusive), consistent with
    ``PiecewiseLinearFunction``.

    ``len(xs) == len(ys) + 1`` and xs must be strictly increasing.

    Example::

        f = StepFunction([0, 3, 7, 10], [10, 20, 30])
        f(0)   # 10
        f(5)   # 20
        f(10)  # 30  (last piece includes right endpoint)

        # Add as constraint to a model:
        y = f.add_constraint(model, x)
    """

    __slots__ = ("xs", "ys")

    def __init__(self, xs: list[int], ys: list[int]) -> None:
        if len(xs) != len(ys) + 1:
            msg = f"len(xs) must be len(ys) + 1, got {len(xs)} and {len(ys)}"
            raise ValueError(msg)
        if len(ys) < 1:
            msg = "Need at least 1 piece"
            raise ValueError(msg)
        if any(x1 >= x2 for x1, x2 in zip(xs, xs[1:], strict=False)):
            msg = "xs must be strictly increasing"
            raise ValueError(msg)
        self.xs: list[int] = list(xs)
        self.ys: list[int] = list(ys)

    @classmethod
    def from_intervals(
        cls,
        intervals: typing.Iterable[tuple[int, int]],
        x_max: int,
    ) -> StepFunction:
        """
        Create from (interval_start, value) pairs plus an upper bound.

        Each pair ``(start, value)`` defines a piece from ``start`` up to
        the next pair's start (or *x_max*).

        Example::

            f = StepFunction.from_intervals(
                [(0, 10), (3, 20), (7, 30)], x_max=10
            )
            # Equivalent to: StepFunction([0, 3, 7, 10], [10, 20, 30])
        """
        pairs = sorted(intervals)
        xs = [s for s, _ in pairs] + [x_max]
        ys = [v for _, v in pairs]
        return cls(xs, ys)

    def __call__(self, x: int) -> int:
        """Return the function value at x."""
        if not self.is_defined_for(x):
            msg = f"x={x} is outside [{self.xs[0]}, {self.xs[-1]}]"
            raise ValueError(msg)
        i = min(bisect.bisect_right(self.xs, x) - 1, len(self.ys) - 1)
        return self.ys[i]

    def is_defined_for(self, x: int) -> bool:
        """Check whether x is within the domain [xs[0], xs[-1]]."""
        return self.xs[0] <= x <= self.xs[-1]

    @property
    def x_min(self) -> int:
        return self.xs[0]

    @property
    def x_max(self) -> int:
        """Last valid x value (inclusive). Same as xs[-1]."""
        return self.xs[-1]

    def is_monotone(self) -> bool:
        """Check if the function is monotonically non-decreasing or non-increasing."""
        return all(
            y1 <= y2 for y1, y2 in zip(self.ys, self.ys[1:], strict=False)
        ) or all(y1 >= y2 for y1, y2 in zip(self.ys, self.ys[1:], strict=False))

    def __repr__(self) -> str:
        return f"StepFunction(xs={self.xs}, ys={self.ys})"

    def simplified(self) -> StepFunction:
        """Return a copy with adjacent equal-value pieces merged.

        This reduces the number of boolean variables needed in the
        constraint encoding without changing the function's behavior.

        Example::

            f = StepFunction([0, 3, 7, 10], [10, 10, 30])
            g = f.simplified()
            # g.xs == [0, 7, 10], g.ys == [10, 30]
        """
        xs_new = [self.xs[0]]
        ys_new = [self.ys[0]]
        for i in range(1, len(self.ys)):
            if self.ys[i] != ys_new[-1]:
                xs_new.append(self.xs[i])
                ys_new.append(self.ys[i])
        xs_new.append(self.xs[-1])
        return StepFunction(xs_new, ys_new)

    # ------------------------------------------------------------------
    # Constraint construction
    # ------------------------------------------------------------------

    def add_constraint(
        self,
        model: cp_model.CpModel,
        x: cp_model.IntVar,
        *,
        y: cp_model.IntVar | None = None,
        name: str = "y",
        restrict_domain: bool = False,
    ) -> cp_model.IntVar:
        """
        Add ``y == f(x)`` constraints to *model*.

        Uses an ordered-step encoding: boolean variables represent whether
        each successive "step" has been taken. Adjacent pieces with equal
        values are merged automatically to reduce boolean variables.

        Args:
            model: The CP-SAT model.
            x: The input integer variable.
            y: Optional output variable. Created automatically if omitted.
            name: Name prefix for created variables (default ``"y"``).
                Use distinct names when adding multiple piecewise
                constraints to the same model.
            restrict_domain: If True, restrict y's domain to the exact set
                of function values (may help propagation).

        Returns:
            The output variable *y* (created or passed through).
        """
        # Merge adjacent equal-value pieces to reduce booleans
        f = self.simplified()

        # Create or accept y variable
        if y is None:
            if restrict_domain:
                domain = cp_model.Domain.from_values(sorted(set(f.ys)))
                y = model.new_int_var_from_domain(domain, name)
            else:
                y = model.new_int_var(min(f.ys), max(f.ys), name)

        # Restrict x to the function's domain (inclusive)
        model.add(x >= f.xs[0])
        model.add(x <= f.xs[-1])

        # Single piece: no booleans needed
        if len(f.ys) == 1:
            model.add(y == f.ys[0])
            return y

        # Step variables: step[i] = 1 means we are in piece i+1 or later
        n_steps = len(f.ys) - 1
        steps = [model.new_bool_var(f"{name}_pcf_step_{i}") for i in range(n_steps)]

        # Steps are ordered: step[i] >= step[i+1]
        for i in range(n_steps - 1):
            model.add(steps[i] >= steps[i + 1])

        # y = ys[0] + sum(step[i] * (ys[i+1] - ys[i]))
        y_expr = f.ys[0] + sum(s * (f.ys[i + 1] - f.ys[i]) for i, s in enumerate(steps))
        model.add(y == y_expr)

        # Link steps to x boundaries:
        # x >= xs[0] + sum(step[i] * (xs[i+1] - xs[i]))
        model.add(
            x >= f.xs[0] + sum(s * (f.xs[i + 1] - f.xs[i]) for i, s in enumerate(steps))
        )
        # x + 1 <= xs[1] + sum(step[i] * (xs[i+2] - xs[i+1])) + steps[-1]
        # The +steps[-1] makes the last piece's upper bound inclusive:
        # for intermediate pieces: x < xs[k+1] (half-open)
        # for the last piece: x <= xs[-1] (closed)
        model.add(
            x + 1
            <= f.xs[1]
            + sum(s * (f.xs[i + 2] - f.xs[i + 1]) for i, s in enumerate(steps))
            + steps[-1]
        )

        return y
