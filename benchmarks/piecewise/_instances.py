"""
Random instance generators for piecewise linear benchmarks.

Contains dataclasses for three problem types (knapsack, production,
dispatch) and functions to generate random instances with configurable
curve shapes.

Usage:
    Internal module — called by __main__.py.

When to modify:
    - To add new problem types
    - To adjust instance difficulty
"""

from __future__ import annotations

import dataclasses
import random

from cpsat_utils.piecewise import PiecewiseLinearFunction

# ---------------------------------------------------------------------------
# Curve generators
# ---------------------------------------------------------------------------


def random_concave_curve(
    x_max: int, n_breakpoints: int, y_scale: int, rng: random.Random
) -> PiecewiseLinearFunction:
    """Generate a concave (diminishing returns) PWL curve."""
    xs = sorted(rng.sample(range(1, x_max), n_breakpoints - 2))
    xs = [0, *xs, x_max]
    gradients = sorted(
        [rng.uniform(0.5, 5.0) for _ in range(len(xs) - 1)], reverse=True
    )
    ys = [0]
    for i, g in enumerate(gradients):
        ys.append(ys[-1] + int(g * (xs[i + 1] - xs[i])))
    scale = y_scale / max(1, max(ys))
    ys = [int(y * scale) for y in ys]
    return PiecewiseLinearFunction(xs, ys)


def random_nonconvex_curve(
    x_max: int, n_breakpoints: int, y_scale: int, rng: random.Random
) -> PiecewiseLinearFunction:
    """Generate a non-convex PWL curve (random gradients)."""
    xs = sorted(rng.sample(range(1, x_max), n_breakpoints - 2))
    xs = [0, *xs, x_max]
    ys = [0]
    for i in range(len(xs) - 1):
        delta = int(rng.uniform(-2, 5) * (xs[i + 1] - xs[i]))
        ys.append(max(0, ys[-1] + delta))
    scale = y_scale / max(1, max(ys))
    ys = [int(y * scale) for y in ys]
    return PiecewiseLinearFunction(xs, ys)


def random_convex_increasing_curve(
    x_min: int,
    x_max: int,
    n_breakpoints: int,
    y_scale: int,
    rng: random.Random,
) -> PiecewiseLinearFunction:
    """Generate a convex increasing PWL curve (for costs)."""
    xs = sorted(rng.sample(range(x_min + 1, x_max), n_breakpoints - 2))
    xs = [x_min, *xs, x_max]
    gradients = sorted([rng.uniform(0.5, 5.0) for _ in range(len(xs) - 1)])
    ys = [0]
    for i, g in enumerate(gradients):
        ys.append(ys[-1] + int(g * (xs[i + 1] - xs[i])))
    scale = y_scale / max(1, max(ys))
    ys = [int(y * scale) for y in ys]
    return PiecewiseLinearFunction(xs, ys)


# ---------------------------------------------------------------------------
# Problem instances
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class KnapsackInstance:
    """Items with piecewise linear value(quantity) curves, shared capacity."""

    value_curves: list[PiecewiseLinearFunction]
    weights: list[int]  # weight per unit of each item
    capacity: int


@dataclasses.dataclass
class ProductionInstance:
    """Products with PWL revenue curves sharing limited resources."""

    revenue_curves: list[PiecewiseLinearFunction]
    resource_usage: list[list[int]]  # [product][resource]
    resource_capacity: list[int]


@dataclasses.dataclass
class DispatchInstance:
    """Generators with PWL cost curves; meet a demand target."""

    cost_curves: list[PiecewiseLinearFunction]  # cost(output)
    min_output: list[int]
    max_output: list[int]
    demand: int


# ---------------------------------------------------------------------------
# Instance generators
# ---------------------------------------------------------------------------


def generate_knapsack(
    n_items: int,
    max_quantity: int,
    n_breakpoints: int,
    capacity_fraction: float = 0.4,
    convex: bool = True,
    seed: int = 42,
) -> KnapsackInstance:
    rng = random.Random(seed)
    curves = []
    weights = []
    for _ in range(n_items):
        if convex:
            f = random_concave_curve(max_quantity, n_breakpoints, 1000, rng)
        else:
            f = random_nonconvex_curve(max_quantity, n_breakpoints, 1000, rng)
        curves.append(f)
        weights.append(rng.randint(1, 10))
    capacity = int(capacity_fraction * n_items * max_quantity)
    return KnapsackInstance(curves, weights, capacity)


def generate_production(
    n_products: int,
    n_resources: int,
    max_production: int,
    n_breakpoints: int,
    convex: bool = True,
    seed: int = 42,
) -> ProductionInstance:
    rng = random.Random(seed)
    curves = []
    usage = []
    for _ in range(n_products):
        if convex:
            f = random_concave_curve(max_production, n_breakpoints, 10000, rng)
        else:
            f = random_nonconvex_curve(max_production, n_breakpoints, 10000, rng)
        curves.append(f)
        usage.append([rng.randint(1, 5) for _ in range(n_resources)])
    capacity = [
        int(0.6 * sum(usage[p][r] * max_production for p in range(n_products)))
        for r in range(n_resources)
    ]
    return ProductionInstance(curves, usage, capacity)


def generate_dispatch(
    n_generators: int,
    max_output: int,
    n_breakpoints: int,
    demand_fraction: float = 0.6,
    seed: int = 42,
) -> DispatchInstance:
    rng = random.Random(seed)
    curves = []
    mins = []
    maxs = []
    for _ in range(n_generators):
        p_min = rng.randint(int(0.1 * max_output), int(0.3 * max_output))
        p_max = rng.randint(int(0.7 * max_output), max_output)
        f = random_convex_increasing_curve(p_min, p_max, n_breakpoints, 5000, rng)
        curves.append(f)
        mins.append(p_min)
        maxs.append(p_max)
    demand = int(demand_fraction * sum(maxs))
    return DispatchInstance(curves, mins, maxs, demand)
