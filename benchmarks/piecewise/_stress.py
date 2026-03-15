"""
Stress test for the best-performing PWL encoding strategies.

Scales up instance sizes to find the limits of `bound+opt+env` and `eq+env`.
Runs each configuration with a single seed and 30s time limit to identify
where solvers start struggling.

Usage:
    python -m benchmarks.piecewise._stress

When to modify:
    - To adjust the scaling ranges
    - To test additional strategies
"""

from __future__ import annotations

from benchmarks.piecewise._instances import (
    generate_dispatch,
    generate_knapsack,
    generate_production,
)
from benchmarks.piecewise._solvers import (
    solve_dispatch,
    solve_knapsack,
    solve_production,
)

STRATEGIES = ["bound+opt+env", "eq+env"]
TIME_LIMIT = 30.0


def _run(label: str, generate, solve, configs: list[dict]) -> None:
    print(f"\n{'=' * 90}")
    print(f"Stress test: {label}")
    print(f"{'=' * 90}")
    print(
        f"{'config':<45} | {'strategy':>14} | "
        f"{'status':>10} {'time_s':>8} {'bools':>7} {'branches':>10}"
    )
    print("-" * 90)
    for config in configs:
        config_str = ", ".join(f"{k}={v}" for k, v in config.items())
        if len(config_str) > 43:
            config_str = config_str[:40] + "..."
        for strategy in STRATEGIES:
            inst = generate(**config, seed=0)
            result = solve(inst, TIME_LIMIT, strategy=strategy)
            status = result["status"]
            marker = " ***" if status != "OPTIMAL" else ""
            print(
                f"{config_str:<45} | {strategy:>14} | "
                f"{status:>10} {result['time_s']:>8.3f} "
                f"{result['num_booleans']:>7} {result['num_branches']:>10}"
                f"{marker}"
            )
        print()


def main():
    # --- Knapsack: push to 1000+ items ---
    _run(
        "Knapsack (convex) — scaling items",
        generate_knapsack,
        solve_knapsack,
        [
            {"n_items": 500, "max_quantity": 100, "n_breakpoints": 10, "convex": True},
            {"n_items": 1000, "max_quantity": 100, "n_breakpoints": 10, "convex": True},
            {"n_items": 2000, "max_quantity": 100, "n_breakpoints": 10, "convex": True},
            {"n_items": 5000, "max_quantity": 100, "n_breakpoints": 10, "convex": True},
        ],
    )

    _run(
        "Knapsack (convex) — scaling breakpoints",
        generate_knapsack,
        solve_knapsack,
        [
            {"n_items": 200, "max_quantity": 100, "n_breakpoints": 50, "convex": True},
            {"n_items": 200, "max_quantity": 1000, "n_breakpoints": 50, "convex": True},
            {"n_items": 200, "max_quantity": 100, "n_breakpoints": 90, "convex": True},
        ],
    )

    _run(
        "Knapsack (non-convex) — scaling",
        generate_knapsack,
        solve_knapsack,
        [
            {"n_items": 500, "max_quantity": 100, "n_breakpoints": 10, "convex": False},
            {
                "n_items": 1000,
                "max_quantity": 100,
                "n_breakpoints": 10,
                "convex": False,
            },
            {"n_items": 200, "max_quantity": 100, "n_breakpoints": 50, "convex": False},
            {"n_items": 200, "max_quantity": 100, "n_breakpoints": 90, "convex": False},
        ],
    )

    # --- Production: the bottleneck — probe resources and products ---
    _run(
        "Production (convex) — scaling products",
        generate_production,
        solve_production,
        [
            {
                "n_products": 100,
                "n_resources": 5,
                "max_production": 100,
                "n_breakpoints": 10,
                "convex": True,
            },
            {
                "n_products": 150,
                "n_resources": 5,
                "max_production": 100,
                "n_breakpoints": 10,
                "convex": True,
            },
            {
                "n_products": 200,
                "n_resources": 5,
                "max_production": 100,
                "n_breakpoints": 10,
                "convex": True,
            },
            {
                "n_products": 300,
                "n_resources": 5,
                "max_production": 100,
                "n_breakpoints": 10,
                "convex": True,
            },
        ],
    )

    _run(
        "Production (convex) — scaling resources",
        generate_production,
        solve_production,
        [
            {
                "n_products": 50,
                "n_resources": 5,
                "max_production": 100,
                "n_breakpoints": 10,
                "convex": True,
            },
            {
                "n_products": 50,
                "n_resources": 10,
                "max_production": 100,
                "n_breakpoints": 10,
                "convex": True,
            },
            {
                "n_products": 50,
                "n_resources": 20,
                "max_production": 100,
                "n_breakpoints": 10,
                "convex": True,
            },
            {
                "n_products": 50,
                "n_resources": 50,
                "max_production": 100,
                "n_breakpoints": 10,
                "convex": True,
            },
        ],
    )

    _run(
        "Production (non-convex) — scaling",
        generate_production,
        solve_production,
        [
            {
                "n_products": 100,
                "n_resources": 5,
                "max_production": 100,
                "n_breakpoints": 10,
                "convex": False,
            },
            {
                "n_products": 150,
                "n_resources": 5,
                "max_production": 100,
                "n_breakpoints": 10,
                "convex": False,
            },
            {
                "n_products": 200,
                "n_resources": 5,
                "max_production": 100,
                "n_breakpoints": 10,
                "convex": False,
            },
            {
                "n_products": 50,
                "n_resources": 10,
                "max_production": 100,
                "n_breakpoints": 10,
                "convex": False,
            },
        ],
    )

    # --- Dispatch: push to 1000+ generators ---
    _run(
        "Dispatch — scaling generators",
        generate_dispatch,
        solve_dispatch,
        [
            {"n_generators": 500, "max_output": 500, "n_breakpoints": 10},
            {"n_generators": 1000, "max_output": 500, "n_breakpoints": 10},
            {"n_generators": 2000, "max_output": 500, "n_breakpoints": 10},
            {"n_generators": 5000, "max_output": 500, "n_breakpoints": 10},
        ],
    )

    _run(
        "Dispatch — scaling breakpoints",
        generate_dispatch,
        solve_dispatch,
        [
            {"n_generators": 200, "max_output": 500, "n_breakpoints": 50},
            {"n_generators": 200, "max_output": 500, "n_breakpoints": 90},
            {"n_generators": 500, "max_output": 500, "n_breakpoints": 50},
        ],
    )


if __name__ == "__main__":
    main()
