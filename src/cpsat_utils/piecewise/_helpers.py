"""
Geometry and arithmetic helpers for piecewise linear constraints.

Contains pure functions for manipulating piecewise linear functions:
collinearity checks, simplification, convex envelope computation,
convex partitioning, and integer line coefficient scaling.

Usage:
    These are internal helpers used by _linear.py and _constraints.py.
    Not part of the public API.

When to modify:
    - To add alternative partitioning strategies
    - To improve convex envelope computation
"""

from __future__ import annotations

import math
import typing

if typing.TYPE_CHECKING:
    from cpsat_utils.piecewise._linear import PiecewiseLinearFunction


def _are_collinear(
    p0: tuple[int, int], p1: tuple[int, int], p2: tuple[int, int]
) -> bool:
    """Check collinearity via cross product (no floating-point error)."""
    return (p1[1] - p0[1]) * (p2[0] - p1[0]) == (p2[1] - p1[1]) * (p1[0] - p0[0])


def _simplify(f: PiecewiseLinearFunction) -> PiecewiseLinearFunction:
    """Remove redundant collinear interior points."""
    from cpsat_utils.piecewise._linear import PiecewiseLinearFunction

    keep = [0]
    for i in range(1, len(f.xs) - 1):
        if not _are_collinear(
            (f.xs[i - 1], f.ys[i - 1]),
            (f.xs[i], f.ys[i]),
            (f.xs[i + 1], f.ys[i + 1]),
        ):
            keep.append(i)
    keep.append(len(f.xs) - 1)
    return PiecewiseLinearFunction([f.xs[i] for i in keep], [f.ys[i] for i in keep])


def _convex_envelope(
    f: PiecewiseLinearFunction, bound_type: str = "upper"
) -> PiecewiseLinearFunction:
    """
    Compute the tightest convex function bounding *f*.

    Uses scipy.spatial.ConvexHull when available; falls back to an
    incremental algorithm otherwise.
    """
    _validate_bound_type(bound_type)
    f = _simplify(f)
    if f.is_convex(bound_type):
        return f.copy()

    try:
        return _convex_envelope_scipy(f, bound_type)
    except ImportError:
        return _convex_envelope_fallback(f, bound_type)


def _convex_envelope_scipy(
    f: PiecewiseLinearFunction, bound_type: str
) -> PiecewiseLinearFunction:
    from scipy.spatial import ConvexHull

    from cpsat_utils.piecewise._linear import PiecewiseLinearFunction

    # Add two dummy points far below (for upper) or above (for lower)
    # so the convex hull only picks boundary points on the correct side.
    dummy_y = min(f.ys) - 1 if bound_type == "upper" else max(f.ys) + 1
    points = list(zip(f.xs, f.ys, strict=True)) + [
        (f.xs[0], dummy_y),
        (f.xs[-1], dummy_y),
    ]
    hull = ConvexHull(points)
    # Keep only original breakpoints that are hull vertices
    vertex_set = set(hull.vertices)
    xs = [f.xs[i] for i in range(len(f.xs)) if i in vertex_set]
    ys = [f.ys[i] for i in range(len(f.ys)) if i in vertex_set]
    result = PiecewiseLinearFunction(xs, ys)
    if not result.is_convex(bound_type):
        msg = f"Convex envelope (scipy) is not convex for bound_type={bound_type!r}"
        raise RuntimeError(msg)
    return result


def _convex_envelope_fallback(
    f: PiecewiseLinearFunction, bound_type: str
) -> PiecewiseLinearFunction:
    """Graham-scan style convex envelope without scipy."""
    from cpsat_utils.piecewise._linear import PiecewiseLinearFunction

    points = list(zip(f.xs, f.ys, strict=True))
    if bound_type == "lower":
        # For lower bound, flip y, compute upper hull, flip back
        points = [(x, -y) for x, y in points]

    # Upper hull: iterate left to right, keep only left turns
    hull: list[tuple[int, int]] = []
    for p in points:
        while len(hull) >= 2:
            # Cross product to check turn direction
            o, a = hull[-2], hull[-1]
            cross = (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0])
            if cross >= 0:  # right turn or collinear — remove
                hull.pop()
            else:
                break
        hull.append(p)

    if bound_type == "lower":
        hull = [(x, -y) for x, y in hull]

    result = PiecewiseLinearFunction([x for x, _ in hull], [y for _, y in hull])
    if not result.is_convex(bound_type):
        msg = f"Convex envelope (fallback) is not convex for bound_type={bound_type!r}"
        raise RuntimeError(msg)
    return result


def _split_into_convex_parts(
    f: PiecewiseLinearFunction, bound_type: str = "upper"
) -> list[PiecewiseLinearFunction]:
    """Partition f into the fewest convex pieces."""
    from cpsat_utils.piecewise._linear import PiecewiseLinearFunction

    _validate_bound_type(bound_type)
    f = _simplify(f)
    if f.is_convex(bound_type):
        return [f.copy()]

    parts: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []

    for x, y in zip(f.xs, f.ys, strict=True):
        if len(current) < 2:
            current.append((x, y))
            continue
        # Compare gradients via integer cross-product (exact, no float).
        # prev_grad = dy1/dx1, curr_grad = dy2/dx2
        # prev >= curr  ⟺  dy1*dx2 >= dy2*dx1  (dx always positive)
        dy1 = current[-1][1] - current[-2][1]
        dx1 = current[-1][0] - current[-2][0]
        dy2 = y - current[-1][1]
        dx2 = x - current[-1][0]
        convex = (
            dy1 * dx2 >= dy2 * dx1 if bound_type == "upper" else dy1 * dx2 <= dy2 * dx1
        )
        if convex:
            current.append((x, y))
        else:
            parts.append(current)
            current = [current[-1], (x, y)]
    if current:
        parts.append(current)

    result = [
        PiecewiseLinearFunction([x for x, _ in p], [y for _, y in p]) for p in parts
    ]
    # Remove trivial single-unit segments that are already bounded by neighbours
    return _remove_redundant_parts(result)


def _remove_redundant_parts(
    parts: list[PiecewiseLinearFunction],
) -> list[PiecewiseLinearFunction]:
    """Drop single-unit interior segments whose endpoints are already constrained."""
    if len(parts) < 3:
        return parts
    redundant: set[int] = set()
    for i in range(1, len(parts) - 1):
        if i - 1 in redundant:
            continue
        p = parts[i]
        if p.xs[-1] - p.xs[0] == 1:
            redundant.add(i)
    return [p for i, p in enumerate(parts) if i not in redundant]


def _integer_line_coefficients(
    x0: int, y0: int, x1: int, y1: int
) -> tuple[int, int, int]:
    """
    Convert a line through (x0, y0) and (x1, y1) into integer coefficients.

    Returns (t, a, b) such that  t * y == a * x + b  on the line,
    where t, a, b are integers with t > 0.
    """
    dy = y1 - y0
    dx = x1 - x0
    if dy != 0:
        lcm = math.lcm(abs(dy), abs(dx))
        t = lcm // abs(dy)
        if t <= 0:
            msg = (
                f"Internal error: expected t > 0, got {t} "
                f"for line ({x0},{y0})->({x1},{y1})"
            )
            raise RuntimeError(msg)
        a = (dy * t) // dx
        if (dy * t) % dx != 0:
            msg = f"Internal error: (dy*t) % dx != 0 for line ({x0},{y0})->({x1},{y1})"
            raise RuntimeError(msg)
    else:
        t = 1
        a = 0
    b = y0 * t - a * x0
    return (t, a, b)


def _validate_bound_type(bound_type: str) -> None:
    if bound_type not in ("upper", "lower"):
        msg = f"bound_type must be 'upper' or 'lower', got {bound_type!r}"
        raise ValueError(msg)
