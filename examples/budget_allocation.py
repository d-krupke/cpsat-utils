"""
Example: Marketing budget allocation with diminishing returns.

A company allocates budget across 4 marketing channels. Each channel has
a piecewise linear response curve: more spend yields more conversions,
but with diminishing returns. A shared budget constraint forces trade-offs.

This is the simplest example of PiecewiseLinearFunction — just curves,
a budget, and a maximize call. No booleans, no time periods, no
indicator constraints.

Note: A pure budget allocation like this is better solved with dedicated
non-linear or LP solvers. CP-SAT shines when piecewise functions appear
inside larger models with combinatorial constraints. This example focuses
on the API, not on choosing the right solver.

Demonstrates:
- PiecewiseLinearFunction construction (from breakpoints and from_function)
- add_upper_bound for maximization (y <= f(x), optimizer pushes y up)
- Reading the optimal solution

Next steps:
- examples/pricing_tiers.py      — StepFunction basics
- examples/energy_dispatch.py    — advanced: both function types combined

Usage:
    python examples/budget_allocation.py
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
from ortools.sat.python import cp_model

from cpsat_utils.piecewise import PiecewiseLinearFunction

# -----------------------------------------------------------------------
# Problem data
# -----------------------------------------------------------------------

BUDGET = 500  # total budget in $1000s

CHANNELS = {
    "Social Media": {
        # Fast initial growth, saturates quickly
        "curve": PiecewiseLinearFunction(
            xs=[0, 50, 100, 150, 200],
            ys=[0, 300, 450, 500, 520],
        ),
        "color": "#1DA1F2",
    },
    "Search Ads": {
        # Steady growth, expensive at scale
        "curve": PiecewiseLinearFunction(
            xs=[0, 50, 100, 200, 300],
            ys=[0, 200, 380, 600, 700],
        ),
        "color": "#34A853",
    },
    "TV": {
        # Needs minimum spend to be effective, then strong growth
        "curve": PiecewiseLinearFunction(
            xs=[0, 50, 100, 150, 250],
            ys=[0, 20, 150, 400, 650],
        ),
        "color": "#FF6B6B",
    },
    "Email": {
        # Cheap and effective, but ceiling is low
        # Built from a function — shows the from_function constructor
        "curve": PiecewiseLinearFunction.from_function(
            lambda x: int(200 * (1 - math.exp(-x / 30))),
            x_min=0,
            x_max=100,
            num_breakpoints=6,
        ),
        "color": "#9B59B6",
    },
}

# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------

model = cp_model.CpModel()

spends = {}
conversions = {}
total_conversions = 0

for name, channel in CHANNELS.items():
    curve = channel["curve"]

    # How much to spend on this channel
    x = model.new_int_var(0, curve.x_max, f"spend_{name}")

    # Conversions from this channel (upper bound — optimizer pushes up)
    y = curve.add_upper_bound(model, x, name=f"conv_{name}")

    spends[name] = x
    conversions[name] = y
    total_conversions += y

# Shared budget
model.add(sum(spends.values()) <= BUDGET)

# Maximize total conversions
model.maximize(total_conversions)

# -----------------------------------------------------------------------
# Solve and print
# -----------------------------------------------------------------------

solver = cp_model.CpSolver()
status = solver.solve(model)
assert status == cp_model.OPTIMAL

print(f"Total budget: ${BUDGET}k")
print(f"Total conversions: {solver.objective_value:.0f}")
print(f"{'':=<50}")
print(f"{'Channel':<15} {'Spend':>8} {'Conversions':>13}")
print(f"{'-' * 50}")
for name in CHANNELS:
    s = solver.value(spends[name])
    c = solver.value(conversions[name])
    print(f"{name:<15} ${s:>6}k {c:>10}")

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle(
    "Marketing Budget Allocation — Diminishing Returns",
    fontsize=14,
    fontweight="bold",
)

for ax, (name, channel) in zip(axes.flat, CHANNELS.items(), strict=False):
    curve = channel["curve"]
    color = channel["color"]

    # Draw the piecewise linear curve
    ax.plot(
        curve.xs,
        curve.ys,
        "-o",
        color=color,
        linewidth=2,
        markersize=4,
        label="Response curve",
    )
    ax.fill_between(curve.xs, curve.ys, alpha=0.1, color=color)

    # Mark the optimal point
    s = solver.value(spends[name])
    c = solver.value(conversions[name])
    ax.plot(
        s, c, "D", color="red", markersize=10, zorder=5, label=f"Optimal: ${s}k → {c}"
    )
    ax.axvline(s, color="red", linestyle=":", alpha=0.4)
    ax.axhline(c, color="red", linestyle=":", alpha=0.4)

    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Spend ($1000s)")
    ax.set_ylabel("Conversions")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
