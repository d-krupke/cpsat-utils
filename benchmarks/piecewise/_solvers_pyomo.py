"""
Pyomo+HiGHS solvers for the same benchmark instances.

Mirrors _solvers.py but uses Pyomo's built-in Piecewise component with
the HiGHS MIP solver, providing a reference implementation for comparing
objective values and solve times against the CP-SAT encodings.

Usage:
    Internal module — called by compare_pyomo.py.

When to modify:
    - To try alternative Pyomo Piecewise formulations (SOS2, CC, MC, etc.)
    - To add new problem solvers matching _solvers.py
"""

from __future__ import annotations

import logging
import time

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    Integers,
    Objective,
    Piecewise,
    Reals,
    SolverFactory,
    TerminationCondition,
    Var,
    maximize,
    minimize,
    value,
)

from benchmarks.piecewise._instances import (
    DispatchInstance,
    KnapsackInstance,
    ProductionInstance,
)

# Suppress noisy Pyomo piecewise warnings about near-equal slopes
logging.getLogger("pyomo.core").setLevel(logging.ERROR)


def _solve_and_report(model: ConcreteModel, time_limit: float) -> dict:
    solver = SolverFactory("appsi_highs")
    solver.options["time_limit"] = time_limit
    solver.options["output_flag"] = False
    t0 = time.perf_counter()
    result = solver.solve(model)
    elapsed = time.perf_counter() - t0
    tc = result.solver.termination_condition
    is_feasible = tc in (TerminationCondition.optimal, TerminationCondition.feasible)
    obj = value(model.obj) if is_feasible else None
    status_name = str(tc)
    return {
        "status": status_name,
        "objective": obj,
        "best_bound": None,
        "time_s": round(elapsed, 3),
    }


def solve_knapsack(
    inst: KnapsackInstance,
    time_limit: float,
) -> dict:
    """Maximize total value subject to shared weight capacity."""
    m = ConcreteModel()
    n = len(inst.value_curves)
    items = range(n)

    # Variables
    m.q = Var(
        items, within=Integers, bounds=lambda _, i: (0, inst.value_curves[i].x_max)
    )
    m.v = Var(
        items,
        within=Reals,
        bounds=lambda _, i: (
            min(inst.value_curves[i].ys),
            max(inst.value_curves[i].ys),
        ),
    )

    # Piecewise constraints: v[i] <= f_i(q[i]) (upper bound; objective pushes v up)
    m.pwl = Piecewise(
        items,
        m.v,
        m.q,
        pw_pts={i: inst.value_curves[i].xs for i in items},
        f_rule={i: inst.value_curves[i].ys for i in items},
        pw_constr_type="UB",
        pw_repn="DCC",
    )

    # Capacity
    m.capacity = Constraint(
        expr=sum(m.q[i] * inst.weights[i] for i in items) <= inst.capacity
    )

    m.obj = Objective(expr=sum(m.v[i] for i in items), sense=maximize)
    return _solve_and_report(m, time_limit)


def solve_production(
    inst: ProductionInstance,
    time_limit: float,
) -> dict:
    """Maximize total revenue subject to shared resource constraints."""
    m = ConcreteModel()
    n_products = len(inst.revenue_curves)
    n_resources = len(inst.resource_capacity)
    products = range(n_products)
    resources = range(n_resources)

    m.q = Var(
        products, within=Integers, bounds=lambda _, i: (0, inst.revenue_curves[i].x_max)
    )
    m.rev = Var(
        products,
        within=Reals,
        bounds=lambda _, i: (
            min(inst.revenue_curves[i].ys),
            max(inst.revenue_curves[i].ys),
        ),
    )

    # Upper bound: rev[i] <= f_i(q[i]); objective pushes rev up
    m.pwl = Piecewise(
        products,
        m.rev,
        m.q,
        pw_pts={i: inst.revenue_curves[i].xs for i in products},
        f_rule={i: inst.revenue_curves[i].ys for i in products},
        pw_constr_type="UB",
        pw_repn="DCC",
    )

    @m.Constraint(resources)
    def resource_limit(_, r):
        return (
            sum(m.q[p] * inst.resource_usage[p][r] for p in products)
            <= inst.resource_capacity[r]
        )

    m.obj = Objective(expr=sum(m.rev[i] for i in products), sense=maximize)
    return _solve_and_report(m, time_limit)


def solve_dispatch(
    inst: DispatchInstance,
    time_limit: float,
) -> dict:
    """Minimize total cost subject to demand constraint.

    The cost curves are defined on [min_output, max_output], but generators
    can be off (output=0). We model this by using p_on for the piecewise
    domain and linking p = p_on * on, so Pyomo's Piecewise sees a variable
    whose domain exactly matches the curve.
    """
    m = ConcreteModel()
    n = len(inst.cost_curves)
    gens = range(n)

    m.on = Var(gens, within=Binary)
    # p_on: output when the generator is on (domain matches cost curve)
    m.p_on = Var(
        gens,
        within=Integers,
        bounds=lambda _, i: (inst.min_output[i], inst.max_output[i]),
    )
    # p: actual output (0 when off, p_on when on)
    m.p = Var(gens, within=Integers, bounds=lambda _, i: (0, inst.max_output[i]))
    m.cost_on = Var(
        gens,
        within=Reals,
        bounds=lambda _, i: (min(inst.cost_curves[i].ys), max(inst.cost_curves[i].ys)),
    )
    m.cost = Var(
        gens, within=Reals, bounds=lambda _, i: (0, max(inst.cost_curves[i].ys))
    )

    # Lower bound: cost_on[i] >= f_i(p_on[i]); objective pushes cost down
    m.pwl = Piecewise(
        gens,
        m.cost_on,
        m.p_on,
        pw_pts={i: inst.cost_curves[i].xs for i in gens},
        f_rule={i: inst.cost_curves[i].ys for i in gens},
        pw_constr_type="LB",
        pw_repn="DCC",
    )

    # Link on/off state: p = p_on if on, p = 0 if off
    # cost = cost_on if on, cost = 0 if off
    @m.Constraint(gens)
    def link_p_lb(_, i):
        return m.p[i] >= m.p_on[i] - inst.max_output[i] * (1 - m.on[i])

    @m.Constraint(gens)
    def link_p_ub(_, i):
        return m.p[i] <= m.p_on[i] + inst.max_output[i] * (1 - m.on[i])

    @m.Constraint(gens)
    def link_p_off(_, i):
        return m.p[i] <= inst.max_output[i] * m.on[i]

    @m.Constraint(gens)
    def link_cost_lb(_, i):
        return m.cost[i] >= m.cost_on[i] - max(inst.cost_curves[i].ys) * (1 - m.on[i])

    @m.Constraint(gens)
    def link_cost_ub(_, i):
        return m.cost[i] <= m.cost_on[i] + max(inst.cost_curves[i].ys) * (1 - m.on[i])

    @m.Constraint(gens)
    def link_cost_off(_, i):
        return m.cost[i] <= max(inst.cost_curves[i].ys) * m.on[i]

    m.demand = Constraint(expr=sum(m.p[i] for i in gens) >= inst.demand)
    m.obj = Objective(expr=sum(m.cost[i] for i in gens), sense=minimize)
    return _solve_and_report(m, time_limit)
