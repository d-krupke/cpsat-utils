"""
Internal constraint builders for piecewise linear functions.

Builds CP-SAT constraints for one-sided bounds (upper/lower) and
equality constraints (floor/ceil/round). Used by PiecewiseLinearFunction
methods; not part of the public API.

Usage:
    Called internally by PiecewiseLinearFunction.add_upper_bound() etc.

When to modify:
    - To add new constraint encoding strategies
    - To add new equality modes beyond floor/ceil/round
"""

from __future__ import annotations

import math

from ortools.sat.python import cp_model

from cpsat_utils.piecewise._helpers import (
    _convex_envelope,
    _integer_line_coefficients,
    _simplify,
    _split_into_convex_parts,
    _validate_bound_type,
)

if __import__("typing").TYPE_CHECKING:
    from cpsat_utils.piecewise._linear import PiecewiseLinearFunction


class _PwlConstraintResult:
    """Holds the result of adding a piecewise linear constraint."""

    __slots__ = (
        "y",
        "num_constraints",
        "num_reified_constraints",
        "num_auxiliary_variables",
    )

    def __init__(self, y: cp_model.IntVar) -> None:
        self.y = y
        self.num_constraints = 0
        self.num_reified_constraints = 0
        self.num_auxiliary_variables = 0


def _add_pwl_constraint(
    model: cp_model.CpModel,
    x: cp_model.IntVar,
    f: PiecewiseLinearFunction,
    bound_type: str,
    *,
    y: cp_model.IntVar | None = None,
    name: str = "y",
    add_convex_envelope: bool = True,
    optimize_partition: bool = True,
) -> _PwlConstraintResult:
    """Build the piecewise linear constraint and return a result object."""
    from cpsat_utils.piecewise._linear import PiecewiseLinearFunction

    _validate_bound_type(bound_type)

    # Create or accept y variable
    if y is None:
        y = model.new_int_var(f.y_min, f.y_max, name)

    result = _PwlConstraintResult(y)

    # Restrict x domain
    model.add(x >= f.x_min)
    model.add(x <= f.x_max)

    # Partition
    if optimize_partition:
        parts = _split_into_convex_parts(f, bound_type)
    else:
        parts = [
            PiecewiseLinearFunction([x1, x2], [y1, y2])
            for (x1, y1), (x2, y2) in f.segments()
        ]

    is_upper = bound_type == "upper"

    if len(parts) == 1:
        # Single convex part — no auxiliary variables needed
        for (x0, y0), (x1, y1) in parts[0].segments():
            t, a, b = _integer_line_coefficients(x0, y0, x1, y1)
            if is_upper:
                model.add(y * t <= a * x + b)
            else:
                model.add(y * t >= a * x + b)
            result.num_constraints += 1
    else:
        # Multiple convex parts — need selector variables
        bvars = [model.new_bool_var(f"{name}_pwl_part_{i}") for i in range(len(parts))]
        result.num_auxiliary_variables += len(bvars)
        model.add_exactly_one(bvars)
        result.num_constraints += 1

        for bvar, part in zip(bvars, parts, strict=True):
            for (x0, y0), (x1, y1) in part.segments():
                t, a, b = _integer_line_coefficients(x0, y0, x1, y1)
                if is_upper:
                    model.add(y * t <= a * x + b).only_enforce_if(bvar)
                else:
                    model.add(y * t >= a * x + b).only_enforce_if(bvar)
                result.num_constraints += 1
                result.num_reified_constraints += 1
            model.add(x >= part.x_min).only_enforce_if(bvar)
            model.add(x <= part.x_max).only_enforce_if(bvar)
            result.num_constraints += 2
            result.num_reified_constraints += 2

        # Convex envelope — redundant but helps the solver
        if add_convex_envelope:
            try:
                envelope = _convex_envelope(f, bound_type)
            except ImportError:
                envelope = None
            if envelope is not None:
                for (x0, y0), (x1, y1) in envelope.segments():
                    t, a, b = _integer_line_coefficients(x0, y0, x1, y1)
                    if is_upper:
                        model.add(y * t <= a * x + b)
                    else:
                        model.add(y * t >= a * x + b)
                    result.num_constraints += 1

    return result


# ---------------------------------------------------------------------------
# Equality constraint builder (floor / ceil / round)
# ---------------------------------------------------------------------------


def _equality_bound_type(mode: str) -> str:
    """Return the bound type used for partitioning/envelope in equality mode.

    - floor: y <= f(x) is the tight side → partition/envelope for "upper"
    - ceil: y >= f(x) is the tight side → partition/envelope for "lower"
    - round: either direction works; "upper" by default
    """
    if mode == "ceil":
        return "lower"
    return "upper"


def _add_equality_constraint(
    model: cp_model.CpModel,
    x: cp_model.IntVar,
    f: PiecewiseLinearFunction,
    mode: str,
    *,
    y: cp_model.IntVar | None = None,
    name: str = "y",
    add_convex_envelope: bool = True,
) -> cp_model.IntVar:
    """
    Add y = floor/ceil/round(f(x)) using both-sided constraints per segment.

    Unlike one-sided bounds, equality constraints always need per-segment
    selector variables (the tightened side from non-active segments
    conflicts). The convex envelope is the main optimization: it adds
    redundant global one-sided constraints that help the solver bound y.

    For each segment with scaled line ``t * y == a * x + b``:
    - floor: ``a*x + b - t + 1 <= t*y <= a*x + b``
    - ceil:  ``a*x + b <= t*y <= a*x + b + t - 1``
    - round: ``2*(a*x + b) - t + 1 <= 2*t*y <= 2*(a*x + b) + t``
    """
    # Remove collinear interior points to avoid unnecessary segment selectors.
    f = _simplify(f)

    if y is None:
        if mode == "ceil":
            lo = min(math.ceil(f(xi)) for xi in f.xs)
            hi = max(math.ceil(f(xi)) for xi in f.xs)
        elif mode == "floor":
            lo = min(math.floor(f(xi)) for xi in f.xs)
            hi = max(math.floor(f(xi)) for xi in f.xs)
        else:
            lo = min(round(f(xi)) for xi in f.xs)
            hi = max(round(f(xi)) for xi in f.xs)
        y = model.new_int_var(lo, hi, name)

    # Restrict x domain
    model.add(x >= f.x_min)
    model.add(x <= f.x_max)

    n_seg = f.num_segments
    if n_seg == 1:
        _add_equality_segment(model, x, y, f.xs[0], f.ys[0], f.xs[1], f.ys[1], mode)
    else:
        bvars = [model.new_bool_var(f"{name}_eq_seg_{i}") for i in range(n_seg)]
        model.add_exactly_one(bvars)
        for i, ((x0, y0), (x1, y1)) in enumerate(f.segments()):
            _add_equality_segment(
                model, x, y, x0, y0, x1, y1, mode, enforced_by=bvars[i]
            )
            model.add(x >= x0).only_enforce_if(bvars[i])
            model.add(x <= x1).only_enforce_if(bvars[i])

        # Convex envelope as redundant one-sided constraint
        if add_convex_envelope:
            if mode == "round":
                # Round benefits from both upper and lower envelopes
                _add_equality_envelope(model, x, y, f, mode, "upper")
                _add_equality_envelope(model, x, y, f, mode, "lower")
            else:
                bound_type = _equality_bound_type(mode)
                _add_equality_envelope(model, x, y, f, mode, bound_type)

    return y


def _add_equality_envelope(
    model: cp_model.CpModel,
    x: cp_model.IntVar,
    y: cp_model.IntVar,
    f: PiecewiseLinearFunction,
    mode: str,
    bound_type: str,
) -> None:
    """Add convex envelope as redundant global constraint for equality modes.

    - floor: y <= f(x), so the upper envelope gives valid global upper bounds
    - ceil: y >= f(x), so the lower envelope gives valid global lower bounds
    - round: y ≈ f(x), so the upper envelope gives y <= envelope(x) + 1
      (weaker but still helpful)
    """
    try:
        envelope = _convex_envelope(f, bound_type)
    except ImportError:
        return

    is_upper = bound_type == "upper"
    for (x0, y0), (x1, y1) in envelope.segments():
        t, a, b = _integer_line_coefficients(x0, y0, x1, y1)
        if mode == "floor":
            # y <= f(x), envelope is above f → y <= envelope(x)
            model.add(y * t <= a * x + b)
        elif mode == "ceil":
            # y >= f(x), envelope is below f → y >= envelope(x)
            model.add(y * t >= a * x + b)
        else:  # round
            # y is within 0.5 of f(x). Envelope bounds f, so:
            if is_upper:
                # upper envelope >= f(x) >= y - 0.5
                # → t*y <= a*x + b + ceil(t/2)
                model.add(y * t <= a * x + b + (t + 1) // 2)
            else:
                model.add(y * t >= a * x + b - (t + 1) // 2)


def _add_equality_segment(
    model: cp_model.CpModel,
    x: cp_model.IntVar,
    y: cp_model.IntVar,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    mode: str,
    *,
    enforced_by: cp_model.IntVar | None = None,
) -> None:
    """Add the two-sided constraint for one segment."""
    t, a, b = _integer_line_coefficients(x0, y0, x1, y1)
    # ax + b is the scaled line value: t * f(x) == a * x + b

    if mode == "floor":
        # t*y <= a*x + b  (y <= f(x))
        # t*y >= a*x + b - t + 1  (y > f(x) - 1, tightened for integers)
        upper = model.add(y * t <= a * x + b)
        lower = model.add(y * t >= a * x + b - t + 1)
    elif mode == "ceil":
        # t*y >= a*x + b  (y >= f(x))
        # t*y <= a*x + b + t - 1  (y < f(x) + 1, tightened for integers)
        upper = model.add(y * t <= a * x + b + t - 1)
        lower = model.add(y * t >= a * x + b)
    else:  # round
        # |y - f(x)| <= 0.5, scaled to avoid fractions:
        # 2*t*y <= 2*(a*x + b) + t
        # 2*t*y >= 2*(a*x + b) - t + 1
        upper = model.add(2 * t * y <= 2 * (a * x + b) + t)
        lower = model.add(2 * t * y >= 2 * (a * x + b) - t + 1)

    if enforced_by is not None:
        upper.only_enforce_if(enforced_by)
        lower.only_enforce_if(enforced_by)
