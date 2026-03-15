"""
Example: Order fulfillment with volume-based pricing tiers.

A distributor receives orders from 5 customers. Each customer's unit price
depends on order size (bulk discounts) — modeled as a StepFunction where
the input (order quantity) is a decision variable. The distributor has
limited warehouse stock and must decide how much to allocate to each
customer to maximize revenue.

Note: A standalone allocation problem like this could be solved with simpler
methods. CP-SAT becomes the right choice when step functions appear inside
larger models with combinatorial constraints (scheduling, routing, etc.).
This example focuses on the API.

Demonstrates:
- StepFunction construction and evaluation
- StepFunction.add_constraint where the input is a decision variable
- The effect of discrete price tiers on optimal allocation

Prerequisites:
- examples/budget_allocation.py  — PiecewiseLinearFunction basics

Next step:
- examples/energy_dispatch.py    — advanced: both function types combined

Usage:
    python examples/pricing_tiers.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from ortools.sat.python import cp_model

from cpsat_utils.piecewise import StepFunction

# -----------------------------------------------------------------------
# Problem data
# -----------------------------------------------------------------------

STOCK = 800  # total units in warehouse

# Each customer has a pricing tier structure:
# small orders pay full price, larger orders get discounts.
CUSTOMERS = {
    "Retail Co": {
        # Willing to pay a lot per unit, but only needs small quantities
        "max_demand": 200,
        "price_tiers": StepFunction(
            xs=[0, 1, 50, 100, 200, 201],
            ys=[0, 120, 100, 85, 70],
            # 0: no order. 1-49: $120/unit. 50-99: $100. etc.
        ),
        "color": "#E74C3C",
    },
    "Wholesale A": {
        # Moderate price, wants medium volume
        "max_demand": 300,
        "price_tiers": StepFunction(
            xs=[0, 1, 100, 200, 300, 301],
            ys=[0, 80, 70, 55, 45],
        ),
        "color": "#3498DB",
    },
    "Wholesale B": {
        # Similar to A but different tier breakpoints
        "max_demand": 250,
        "price_tiers": StepFunction(
            xs=[0, 1, 80, 150, 250, 251],
            ys=[0, 90, 75, 60, 50],
        ),
        "color": "#2ECC71",
    },
    "Big Box": {
        # Lowest per-unit price but huge volume
        "max_demand": 500,
        "price_tiers": StepFunction(
            xs=[0, 1, 150, 300, 500, 501],
            ys=[0, 65, 55, 45, 35],
        ),
        "color": "#F39C12",
    },
    "Online": {
        # Premium pricing, moderate demand
        "max_demand": 150,
        "price_tiers": StepFunction(
            xs=[0, 1, 40, 80, 150, 151],
            ys=[0, 130, 110, 95, 80],
        ),
        "color": "#9B59B6",
    },
}

# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------

model = cp_model.CpModel()

quantities = {}
prices = {}
total_revenue = 0

for name, cust in CUSTOMERS.items():
    tiers = cust["price_tiers"]

    # How many units to allocate to this customer
    q = model.new_int_var(0, cust["max_demand"], f"qty_{name}")

    # Price per unit — determined by the step function of quantity
    p = tiers.add_constraint(model, q, name=f"price_{name}")

    # Revenue = quantity * price_per_unit
    # CP-SAT needs a helper variable for the product
    rev = model.new_int_var(0, cust["max_demand"] * max(tiers.ys), f"rev_{name}")
    model.add_multiplication_equality(rev, [q, p])

    quantities[name] = q
    prices[name] = p
    total_revenue += rev

# Total allocation cannot exceed stock
model.add(sum(quantities.values()) <= STOCK)

# Maximize total revenue
model.maximize(total_revenue)

# -----------------------------------------------------------------------
# Solve and print
# -----------------------------------------------------------------------

solver = cp_model.CpSolver()
status = solver.solve(model)
assert status == cp_model.OPTIMAL

print(f"Warehouse stock: {STOCK} units")
print(f"Total revenue: ${solver.objective_value:,.0f}")
print(f"{'':=<60}")
print(f"{'Customer':<14} {'Qty':>5} {'Price':>8} {'Revenue':>10} {'Tier'}")
print(f"{'-' * 60}")
total_alloc = 0
for name, cust in CUSTOMERS.items():
    q = solver.value(quantities[name])
    p = solver.value(prices[name])
    rev = q * p
    total_alloc += q
    tier = f"${p}/unit" if q > 0 else "—"
    print(f"{name:<14} {q:>5} {tier:>8} ${rev:>8,}   ", end="")
    # Show which tier they landed in
    tiers = cust["price_tiers"]
    if q > 0:
        for i in range(len(tiers.ys)):
            if tiers.xs[i] <= q <= tiers.xs[i + 1]:
                lo, hi = tiers.xs[i], tiers.xs[i + 1] - 1
                print(f"({lo}-{hi} units)")
                break
    else:
        print()
print(f"{'-' * 60}")
print(f"{'Total':<14} {total_alloc:>5}")

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle(
    "Volume Pricing Tiers (StepFunction) — Optimal Allocation",
    fontsize=14,
    fontweight="bold",
)

for ax, (name, cust) in zip(axes.flat, CUSTOMERS.items(), strict=False):
    tiers = cust["price_tiers"]
    color = cust["color"]

    # Draw the step function (skip the 0-quantity "no order" tier)
    for i in range(1, len(tiers.ys)):
        x_lo, x_hi = tiers.xs[i], tiers.xs[i + 1]
        ax.fill_between(
            [x_lo, x_hi],
            [tiers.ys[i], tiers.ys[i]],
            alpha=0.2,
            color=color,
        )
        ax.plot(
            [x_lo, x_hi],
            [tiers.ys[i], tiers.ys[i]],
            "-",
            color=color,
            linewidth=2.5,
        )
        if i > 1:
            ax.plot(
                [x_lo, x_lo],
                [tiers.ys[i - 1], tiers.ys[i]],
                ":",
                color=color,
                linewidth=1,
            )

    # Mark optimal quantity
    q = solver.value(quantities[name])
    p = solver.value(prices[name])
    if q > 0:
        ax.plot(q, p, "D", color="red", markersize=10, zorder=5)
        ax.annotate(
            f"  {q} units\n  ${p}/unit\n  ${q * p:,} rev",
            (q, p),
            fontsize=8,
            color="red",
            fontweight="bold",
            va="top",
        )

    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Order Quantity")
    ax.set_ylabel("Price per Unit ($)")
    ax.set_xlim(0, cust["max_demand"] * 1.05)
    ax.grid(True, alpha=0.3)

# Hide unused subplot
axes[1, 2].set_visible(False)

plt.tight_layout()
plt.show()
