"""
Compare CP-SAT piecewise encoding against Pyomo+HiGHS.

Runs each benchmark instance with our best CP-SAT strategy and the
Pyomo Piecewise+HiGHS reference, then compares objective values and
solve times side by side.

Usage:
    python -m benchmarks.piecewise.compare_pyomo [--seeds 3] [--time-limit 30]
                                                  [--problem all]

When to modify:
    - To adjust which CP-SAT strategy is used as "best"
    - To add new problem types
"""

from __future__ import annotations

import argparse
import sys

from benchmarks.piecewise._instances import (
    generate_dispatch,
    generate_knapsack,
    generate_production,
)
from benchmarks.piecewise._solvers import (
    solve_dispatch as cpsat_dispatch,
)
from benchmarks.piecewise._solvers import (
    solve_knapsack as cpsat_knapsack,
)
from benchmarks.piecewise._solvers import (
    solve_production as cpsat_production,
)
from benchmarks.piecewise._solvers_pyomo import (
    solve_dispatch as pyomo_dispatch,
)
from benchmarks.piecewise._solvers_pyomo import (
    solve_knapsack as pyomo_knapsack,
)
from benchmarks.piecewise._solvers_pyomo import (
    solve_production as pyomo_production,
)

# Best CP-SAT strategy: one-sided bound with optimized convex partition
# and convex envelope. Faster than equality because it uses fewer boolean
# variables (convex partition vs per-segment selectors), and the objective
# naturally tightens the bound.
CPSAT_STRATEGY = "bound+opt+env"

PROBLEMS = {
    "knapsack": {
        "generate": generate_knapsack,
        "cpsat_solve": cpsat_knapsack,
        "pyomo_solve": pyomo_knapsack,
        "configs": [
            {"n_items": 100, "max_quantity": 200, "n_breakpoints": 15, "convex": True},
            {"n_items": 100, "max_quantity": 200, "n_breakpoints": 15, "convex": False},
            {"n_items": 200, "max_quantity": 200, "n_breakpoints": 10, "convex": False},
            {"n_items": 500, "max_quantity": 500, "n_breakpoints": 10, "convex": False},
        ],
    },
    "production": {
        "generate": generate_production,
        "cpsat_solve": cpsat_production,
        "pyomo_solve": pyomo_production,
        "configs": [
            {
                "n_products": 30,
                "n_resources": 5,
                "max_production": 200,
                "n_breakpoints": 15,
                "convex": False,
            },
            {
                "n_products": 50,
                "n_resources": 5,
                "max_production": 200,
                "n_breakpoints": 15,
                "convex": True,
            },
            {
                "n_products": 50,
                "n_resources": 5,
                "max_production": 200,
                "n_breakpoints": 15,
                "convex": False,
            },
            {
                "n_products": 80,
                "n_resources": 5,
                "max_production": 200,
                "n_breakpoints": 10,
                "convex": False,
            },
        ],
    },
    "dispatch": {
        "generate": generate_dispatch,
        "cpsat_solve": cpsat_dispatch,
        "pyomo_solve": pyomo_dispatch,
        "configs": [
            {"n_generators": 50, "max_output": 500, "n_breakpoints": 15},
            {"n_generators": 100, "max_output": 1000, "n_breakpoints": 10},
            {"n_generators": 100, "max_output": 1000, "n_breakpoints": 20},
            {"n_generators": 200, "max_output": 1000, "n_breakpoints": 15},
        ],
    },
}


def _header() -> str:
    return f"{'seed':>5} | {'solver':>12} | {'status':>12} {'obj':>10} {'time_s':>8}"


def _format_row(seed: int, solver_name: str, result: dict) -> str:
    obj_str = f"{result['objective']:.0f}" if result["objective"] is not None else "N/A"
    return (
        f"{seed:>5} | {solver_name:>12} | "
        f"{result['status']:>12} {obj_str:>10} {result['time_s']:>8.3f}"
    )


def run_comparison(
    problem: str,
    n_seeds: int = 3,
    time_limit: float = 30.0,
    output=None,
) -> None:
    if output is None:
        output = sys.stdout
    spec = PROBLEMS[problem]
    generate = spec["generate"]
    cpsat_solve = spec["cpsat_solve"]
    pyomo_solve = spec["pyomo_solve"]

    def p(text=""):
        print(text, file=output, flush=True)

    p(f"\n{'=' * 60}")
    p(f"Problem: {problem}")
    p(f"{'=' * 60}")

    mismatches = 0
    total = 0
    max_diffs: list[float] = []

    for config in spec["configs"]:
        p(f"\n--- Config: {config} ---")
        p(_header())
        p("-" * 60)

        for seed in range(n_seeds):
            instance = generate(**config, seed=seed)

            cpsat_result = cpsat_solve(instance, time_limit, strategy=CPSAT_STRATEGY)
            pyomo_result = pyomo_solve(instance, time_limit)

            p(_format_row(seed, f"cpsat({CPSAT_STRATEGY})", cpsat_result))
            p(_format_row(seed, "pyomo+highs", pyomo_result))

            # Compare objectives — CP-SAT uses integer floor/ceil while
            # Pyomo uses continuous interpolation, so differences of ~1 per
            # curve are expected. Flag only large discrepancies.
            total += 1
            c_obj = cpsat_result["objective"]
            p_obj = pyomo_result["objective"]
            if c_obj is not None and p_obj is not None:
                diff = abs(c_obj - p_obj)
                if diff > 0.5:
                    max_diffs.append(diff)
                    if diff > 10:
                        p(
                            f"  *** LARGE MISMATCH: "
                            f"cpsat={c_obj:.0f} vs "
                            f"pyomo={p_obj:.0f} "
                            f"(diff={diff:.0f})"
                        )
                        mismatches += 1
            p("")

    avg_diff = sum(max_diffs) / len(max_diffs) if max_diffs else 0
    max_diff = max(max_diffs) if max_diffs else 0
    p(f"\n--- Summary: {total} instances, {mismatches} large mismatches (>10) ---")
    p(
        f"    Objective diffs >0.5: {len(max_diffs)}/{total}, "
        f"avg={avg_diff:.1f}, max={max_diff:.0f}"
    )
    p(
        "    (Small diffs expected: CP-SAT uses integer ceil/floor, "
        "Pyomo uses continuous interpolation)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare CP-SAT piecewise vs Pyomo+HiGHS"
    )
    parser.add_argument(
        "--problem",
        choices=list(PROBLEMS.keys()) + ["all"],
        default="all",
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--time-limit", type=float, default=30.0)
    args = parser.parse_args()

    problems = list(PROBLEMS.keys()) if args.problem == "all" else [args.problem]
    for problem in problems:
        run_comparison(problem, n_seeds=args.seeds, time_limit=args.time_limit)


if __name__ == "__main__":
    main()
