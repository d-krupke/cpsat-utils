"""
Solver functions for piecewise linear benchmarks.

Each problem has a solve function that accepts an encoding strategy name.
Strategies control how PWL functions are added to the model:

- "bound+opt+env": one-sided bound, optimized partition, convex envelope
- "bound+opt":     one-sided bound, optimized partition, no envelope
- "bound+naive":   one-sided bound, one segment per piece, no envelope
- "equality":      floor/ceil equality constraint (tighter, per-segment)

Usage:
    Internal module — called by __main__.py.

When to modify:
    - To add new encoding strategies
    - To add new problem solvers
"""

from __future__ import annotations

import time

from ortools.sat.python import cp_model

from benchmarks.piecewise._instances import (
    DispatchInstance,
    KnapsackInstance,
    ProductionInstance,
)

STRATEGIES = [
    "bound+opt+env",
    "bound+opt",
    "bound+naive",
    "eq+env",
    "eq+naive",
]


def _bound_kwargs(strategy: str) -> dict:
    """Convert strategy name to keyword arguments for add_upper/lower_bound."""
    if strategy == "bound+opt+env":
        return {"optimize_partition": True, "add_convex_envelope": True}
    if strategy == "bound+opt":
        return {"optimize_partition": True, "add_convex_envelope": False}
    if strategy == "bound+naive":
        return {"optimize_partition": False, "add_convex_envelope": False}
    msg = f"Not a bound strategy: {strategy!r}"
    raise ValueError(msg)


def _equality_kwargs(strategy: str) -> dict:
    """Convert strategy name to keyword arguments for add_floor/ceil/round."""
    if strategy == "eq+env":
        return {"add_convex_envelope": True}
    if strategy == "eq+naive":
        return {"add_convex_envelope": False}
    msg = f"Not an equality strategy: {strategy!r}"
    raise ValueError(msg)


def _is_equality(strategy: str) -> bool:
    return strategy.startswith("eq+")


def solve_knapsack(
    inst: KnapsackInstance,
    time_limit: float,
    strategy: str = "bound+opt+env",
) -> dict:
    """Maximize total value subject to shared weight capacity."""
    model = cp_model.CpModel()
    quantities = []
    total_value = 0
    for i, (curve, _w) in enumerate(zip(inst.value_curves, inst.weights, strict=True)):
        q = model.new_int_var(0, curve.x_max, f"q_{i}")
        quantities.append(q)
        if _is_equality(strategy):
            v = curve.add_floor(model, q, **_equality_kwargs(strategy))
        else:
            v = curve.add_upper_bound(model, q, **_bound_kwargs(strategy))
        total_value += v
    model.add(
        sum(q * w for q, w in zip(quantities, inst.weights, strict=True))
        <= inst.capacity
    )
    model.maximize(total_value)
    return _solve_and_report(model, time_limit)


def solve_production(
    inst: ProductionInstance,
    time_limit: float,
    strategy: str = "bound+opt+env",
) -> dict:
    """Maximize total revenue subject to shared resource constraints."""
    model = cp_model.CpModel()
    n_products = len(inst.revenue_curves)
    n_resources = len(inst.resource_capacity)

    quantities = []
    total_revenue = 0
    for i, curve in enumerate(inst.revenue_curves):
        q = model.new_int_var(0, curve.x_max, f"prod_{i}")
        quantities.append(q)
        if _is_equality(strategy):
            rev = curve.add_floor(model, q, **_equality_kwargs(strategy))
        else:
            rev = curve.add_upper_bound(model, q, **_bound_kwargs(strategy))
        total_revenue += rev

    for r in range(n_resources):
        model.add(
            sum(quantities[p] * inst.resource_usage[p][r] for p in range(n_products))
            <= inst.resource_capacity[r]
        )

    model.maximize(total_revenue)
    return _solve_and_report(model, time_limit)


def solve_dispatch(
    inst: DispatchInstance,
    time_limit: float,
    strategy: str = "bound+opt+env",
) -> dict:
    """Minimize total cost subject to demand constraint."""
    model = cp_model.CpModel()

    outputs = []
    total_cost = 0
    for i, curve in enumerate(inst.cost_curves):
        on = model.new_bool_var(f"on_{i}")
        p = model.new_int_var(0, inst.max_output[i], f"output_{i}")
        outputs.append(p)
        model.add(p >= inst.min_output[i]).only_enforce_if(on)
        model.add(p <= inst.max_output[i]).only_enforce_if(on)
        model.add(p == 0).only_enforce_if(on.negated())
        if _is_equality(strategy):
            cost = curve.add_ceil(model, p, **_equality_kwargs(strategy))
        else:
            cost = curve.add_lower_bound(model, p, **_bound_kwargs(strategy))
        total_cost += cost

    model.add(sum(outputs) >= inst.demand)
    model.minimize(total_cost)
    return _solve_and_report(model, time_limit)


def _solve_and_report(model: cp_model.CpModel, time_limit: float) -> dict:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    t0 = time.perf_counter()
    status = solver.solve(model)
    elapsed = time.perf_counter() - t0
    is_feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    return {
        "status": solver.status_name(status),
        "objective": solver.objective_value if is_feasible else None,
        "best_bound": solver.best_objective_bound if is_feasible else None,
        "time_s": round(elapsed, 3),
        "num_booleans": solver.num_booleans,
        "num_branches": solver.num_branches,
        "num_conflicts": solver.num_conflicts,
    }
