"""
Example: Power plant dispatch with piecewise linear fuel costs and
step-function crew staffing costs.

NOTE: This is the advanced example. If you're new to cpsat_utils, start with:
  - examples/budget_allocation.py  (PiecewiseLinearFunction basics)
  - examples/pricing_tiers.py      (StepFunction basics)

A utility company operates 4 power generators to meet electricity demand
across 6 time periods. Each generator has:
- A piecewise linear fuel cost curve (efficiency drops at high output)
- An on/off decision with minimum output when on

Additionally, operating crew costs follow a step function: the number of
generators running in a period determines how many crews are needed. More
active generators require more crews, with discrete cost jumps.

Note: A pure unit commitment problem is typically solved with MIP solvers.
This example is a good fit for CP-SAT because it combines piecewise costs
with discrete on/off decisions and step-function crew costs — the kind of
hybrid structure where CP-SAT excels.

Demonstrates:
- PiecewiseLinearFunction.add_lower_bound for convex cost minimization
- StepFunction.add_constraint where the input is a decision variable
- Combining both function types in a single model
- Indicator constraints (on/off decisions) with piecewise costs
- Multi-period scheduling with shared resource constraints

Usage:
    python examples/energy_dispatch.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from ortools.sat.python import cp_model

from cpsat_utils.piecewise import PiecewiseLinearFunction, StepFunction

# -----------------------------------------------------------------------
# Problem data
# -----------------------------------------------------------------------

# Four generators with different characteristics
GENERATORS = {
    "Coal A": {
        "cost_curve": PiecewiseLinearFunction(
            xs=[0, 50, 100, 150, 200],
            ys=[0, 400, 900, 1600, 2800],
            # cheap baseload, cost rises sharply at high output
        ),
        "min_output": 30,  # MW
        "max_output": 200,
        "color": "#8B4513",
    },
    "Gas B": {
        "cost_curve": PiecewiseLinearFunction(
            xs=[0, 40, 80, 120, 160],
            ys=[0, 500, 1100, 1800, 2600],
            # moderate cost, flatter curve
        ),
        "min_output": 20,
        "max_output": 160,
        "color": "#FF8C00",
    },
    "Gas C": {
        "cost_curve": PiecewiseLinearFunction(
            xs=[0, 30, 60, 100],
            ys=[0, 450, 1000, 2000],
            # small, expensive peaker
        ),
        "min_output": 15,
        "max_output": 100,
        "color": "#DAA520",
    },
    "Hydro": {
        "cost_curve": PiecewiseLinearFunction(
            xs=[0, 25, 50, 75],
            ys=[0, 50, 120, 250],
            # very cheap but limited capacity
        ),
        "min_output": 5,
        "max_output": 75,
        "color": "#4682B4",
    },
}

# Demand over 6 time periods (4-hour blocks)
PERIODS = ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]
DEMAND = [180, 220, 350, 380, 320, 200]  # MW per period

# Electricity sale price per period ($/MWh) — for profit reporting only.
# Since demand is a hard constraint, revenue is fixed and doesn't affect
# the optimization. The model minimizes cost; profit = revenue - cost.
PRICES = [30, 35, 55, 60, 50, 30]

# Crew staffing cost: depends on how many generators are running.
# 1 gen → 1 crew ($200), 2 gen → $500, 3 gen → $900, 4 gen → $1400
# This is a StepFunction whose input is a decision variable.
CREW_COST = StepFunction(
    xs=[0, 1, 2, 3, 4, 5],
    ys=[0, 200, 500, 900, 1400],
    # 0 gens: no crew. 1: $200. 2: $500. 3: $900. 4: $1400.
)


# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------


def build_and_solve():
    model = cp_model.CpModel()

    gen_names = list(GENERATORS.keys())
    n_gens = len(gen_names)
    outputs = {}  # (gen, period) -> IntVar
    costs = {}  # (gen, period) -> IntVar
    on_vars = {}  # (gen, period) -> BoolVar
    crew_costs = {}  # period -> IntVar

    total_cost = 0

    for t, period in enumerate(PERIODS):
        period_output = []
        period_on = []

        for name in gen_names:
            gen = GENERATORS[name]
            curve = gen["cost_curve"]

            on = model.new_bool_var(f"on_{name}_{period}")
            p = model.new_int_var(0, gen["max_output"], f"p_{name}_{period}")

            # Operating range when on
            model.add(p >= gen["min_output"]).only_enforce_if(on)
            model.add(p <= gen["max_output"]).only_enforce_if(on)
            model.add(p == 0).only_enforce_if(on.negated())

            # Fuel cost (lower bound — minimization pushes to curve)
            cost = curve.add_lower_bound(model, p, name=f"cost_{name}_{period}")

            outputs[name, t] = p
            costs[name, t] = cost
            on_vars[name, t] = on
            period_output.append(p)
            period_on.append(on)
            total_cost += cost

        # Must exactly meet demand
        model.add(sum(period_output) == DEMAND[t])

        # Crew staffing cost — StepFunction of number of active generators
        n_active = model.new_int_var(0, n_gens, f"n_active_{period}")
        model.add(n_active == sum(period_on))
        crew = CREW_COST.add_constraint(model, n_active, name=f"crew_{period}")
        crew_costs[t] = crew
        total_cost += crew

    # Minimize total cost (fuel + crew)
    model.minimize(total_cost)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"No solution found: {solver.status_name(status)}")
        return None

    total_fuel = sum(
        solver.value(costs[name, t]) for name in gen_names for t in range(len(PERIODS))
    )
    total_crew = sum(solver.value(crew_costs[t]) for t in range(len(PERIODS)))
    total_rev = sum(PRICES[t] * DEMAND[t] for t in range(len(PERIODS)))

    solution = {
        "status": solver.status_name(status),
        "total_fuel": total_fuel,
        "total_crew": total_crew,
        "total_cost": total_fuel + total_crew,
        "total_revenue": total_rev,
        "profit": total_rev - total_fuel - total_crew,
        "periods": [],
    }

    print(f"\nStatus: {solver.status_name(status)}")
    print(f"Total fuel cost:  ${total_fuel:>8,}")
    print(f"Total crew cost:  ${total_crew:>8,}")
    print(f"Total cost:       ${total_fuel + total_crew:>8,}")
    print(f"Total revenue:    ${total_rev:>8,}")
    print(f"Profit:           ${total_rev - total_fuel - total_crew:>8,}")
    print(f"{'':=<70}")

    for t, period in enumerate(PERIODS):
        n_on = sum(solver.value(on_vars[name, t]) for name in gen_names)
        crew_val = solver.value(crew_costs[t])
        period_data = {
            "period": period,
            "demand": DEMAND[t],
            "price": PRICES[t],
            "n_active": n_on,
            "crew_cost": crew_val,
            "generators": {},
        }
        print(
            f"\nPeriod {period} | Demand: {DEMAND[t]} MW "
            f"| Price: ${PRICES[t]}/MWh "
            f"| Crew: {n_on} gen → ${crew_val}"
        )
        print(f"  {'Generator':<10} {'Status'}")
        print(f"  {'-' * 40}")

        total_output = 0
        total_period_cost = 0
        for name in gen_names:
            p_val = solver.value(outputs[name, t])
            c_val = solver.value(costs[name, t])
            on_val = solver.value(on_vars[name, t])
            total_output += p_val
            total_period_cost += c_val
            if on_val:
                print(f"  {name:<10} {p_val:>4} MW  ${c_val:>6}")
            else:
                print(f"  {name:<10}      OFF")
            period_data["generators"][name] = {
                "output": p_val,
                "cost": c_val,
                "on": bool(on_val),
            }

        revenue = PRICES[t] * total_output
        period_data["total_output"] = total_output
        period_data["total_cost"] = total_period_cost + crew_val
        period_data["revenue"] = revenue
        solution["periods"].append(period_data)

    return solution


# -----------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------


def plot_cost_curves(solution):
    """Plot generator cost curves with optimal operating points."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Generator Fuel Cost Curves (PiecewiseLinearFunction)",
        fontsize=14,
        fontweight="bold",
    )

    for ax, (name, gen) in zip(axes.flat, GENERATORS.items(), strict=False):
        curve = gen["cost_curve"]
        xs = np.array(curve.xs)
        ys = np.array(curve.ys)
        ax.plot(
            xs,
            ys,
            "-",
            color=gen["color"],
            linewidth=2.5,
            label="Cost curve",
        )
        ax.fill_between(xs, ys, alpha=0.1, color=gen["color"])

        # Operating range
        ax.axvspan(
            gen["min_output"],
            gen["max_output"],
            alpha=0.08,
            color="green",
            label="Operating range",
        )
        ax.axvline(
            gen["min_output"],
            color="green",
            linestyle=":",
            alpha=0.5,
        )
        ax.axvline(
            gen["max_output"],
            color="green",
            linestyle=":",
            alpha=0.5,
        )

        # Optimal operating points
        if solution:
            for period_data in solution["periods"]:
                gdata = period_data["generators"][name]
                if gdata["on"]:
                    ax.plot(
                        gdata["output"],
                        gdata["cost"],
                        "o",
                        color="red",
                        markersize=7,
                        zorder=5,
                    )
                    ax.annotate(
                        period_data["period"],
                        (gdata["output"], gdata["cost"]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=7,
                        color="red",
                    )

        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Output (MW)")
        ax.set_ylabel("Fuel Cost ($/h)")
        ax.set_xlim(0, gen["max_output"] * 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_crew_cost(solution):
    """Plot the crew staffing step function with solution points."""
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        width_ratios=[1, 1.5],
    )
    fig.suptitle(
        "Crew Staffing Cost (StepFunction)",
        fontsize=14,
        fontweight="bold",
    )

    # Left: the step function itself
    xs = CREW_COST.xs
    ys = CREW_COST.ys
    for i in range(len(ys)):
        x_start, x_end = xs[i], xs[i + 1]
        ax1.fill_between(
            [x_start, x_end],
            [ys[i], ys[i]],
            alpha=0.3,
            color="teal",
        )
        ax1.plot(
            [x_start, x_end],
            [ys[i], ys[i]],
            "-",
            color="teal",
            linewidth=2.5,
        )
        if ys[i] > 0:
            ax1.text(
                (x_start + x_end) / 2,
                ys[i] + 30,
                f"${ys[i]}",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
        if i > 0:
            ax1.plot(
                [x_start, x_start],
                [ys[i - 1], ys[i]],
                ":",
                color="teal",
                linewidth=1,
            )

    # Mark solution points
    if solution:
        for period_data in solution["periods"]:
            n = period_data["n_active"]
            c = period_data["crew_cost"]
            ax1.plot(n, c, "o", color="red", markersize=8, zorder=5)
            ax1.annotate(
                period_data["period"],
                (n, c),
                textcoords="offset points",
                xytext=(5, 8),
                fontsize=8,
                color="red",
            )

    ax1.set_xlabel("Active Generators")
    ax1.set_ylabel("Crew Cost ($/period)")
    ax1.set_xticks(range(5))
    ax1.set_ylim(-50, max(ys) * 1.2)
    ax1.grid(True, alpha=0.3)

    # Right: crew cost per period as bar chart
    if solution:
        x = np.arange(len(PERIODS))
        crew = [p["crew_cost"] for p in solution["periods"]]
        n_active = [p["n_active"] for p in solution["periods"]]
        bars = ax2.bar(x, crew, 0.6, color="teal", alpha=0.7)

        for bar, n in zip(bars, n_active, strict=True):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                f"{n} gen",
                ha="center",
                fontsize=9,
            )

        ax2.set_xticks(x)
        ax2.set_xticklabels(PERIODS)
        ax2.set_xlabel("Time of Day")
        ax2.set_ylabel("Crew Cost ($/period)")
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_dispatch_schedule(solution):
    """Stacked bar chart of dispatch schedule across periods."""
    if not solution:
        return None

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        height_ratios=[2, 1],
    )
    fig.suptitle(
        "Optimal Dispatch Schedule",
        fontsize=14,
        fontweight="bold",
    )

    gen_names = list(GENERATORS.keys())
    x = np.arange(len(PERIODS))
    width = 0.6

    # Stacked bars for generator outputs
    bottoms = np.zeros(len(PERIODS))
    for name in gen_names:
        vals = [
            solution["periods"][t]["generators"][name]["output"]
            for t in range(len(PERIODS))
        ]
        ax1.bar(
            x,
            vals,
            width,
            bottom=bottoms,
            label=name,
            color=GENERATORS[name]["color"],
            edgecolor="white",
        )
        bottoms += np.array(vals)

    # Demand line
    ax1.step(
        np.append(x - width / 2, x[-1] + width / 2),
        np.append(DEMAND, DEMAND[-1]),
        where="post",
        color="red",
        linewidth=2,
        linestyle="--",
        label="Demand",
    )

    ax1.set_ylabel("Output (MW)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(PERIODS)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3, axis="y")

    # Cost breakdown per period
    fuel = [
        sum(solution["periods"][t]["generators"][n]["cost"] for n in gen_names)
        for t in range(len(PERIODS))
    ]
    crew = [p["crew_cost"] for p in solution["periods"]]
    revenues = [p["revenue"] for p in solution["periods"]]
    profits = [r - f - c for r, f, c in zip(revenues, fuel, crew, strict=True)]

    ax2.bar(x - 0.15, fuel, 0.3, color="red", alpha=0.4, label="Fuel")
    ax2.bar(
        x - 0.15,
        crew,
        0.3,
        bottom=fuel,
        color="teal",
        alpha=0.4,
        label="Crew",
    )
    ax2.bar(
        x + 0.15,
        profits,
        0.3,
        color="green",
        alpha=0.6,
        label="Profit",
    )

    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        x,
        PRICES,
        "s-",
        color="purple",
        linewidth=2,
        label="Price ($/MWh)",
    )

    ax2.set_ylabel("$/period")
    ax2_twin.set_ylabel("Price ($/MWh)", color="purple")
    ax2.set_xticks(x)
    ax2.set_xticklabels(PERIODS)
    ax2.set_xlabel("Time of Day")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
    )
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    solution = build_and_solve()

    fig1 = plot_cost_curves(solution)
    fig2 = plot_crew_cost(solution)
    fig3 = plot_dispatch_schedule(solution)

    plt.show()
