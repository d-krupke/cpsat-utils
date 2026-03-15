"""
CLI runner for piecewise linear constraint benchmarks.

Compares encoding strategies across three problem types:
    1. Piecewise knapsack — items with PWL value curves, shared capacity
    2. Multi-product production — PWL revenue, shared resources
    3. Generator dispatch — PWL cost curves, meet demand

Usage:
    python -m benchmarks.piecewise [--problem knapsack|production|dispatch|all]
                                   [--seeds 5] [--time-limit 30]

When to modify:
    - To adjust instance configurations
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
    STRATEGIES,
    solve_dispatch,
    solve_knapsack,
    solve_production,
)

PROBLEMS = {
    "knapsack": {
        "generate": generate_knapsack,
        "solve": solve_knapsack,
        "configs": [
            {"n_items": 10, "max_quantity": 100, "n_breakpoints": 5, "convex": True},
            {"n_items": 10, "max_quantity": 100, "n_breakpoints": 5, "convex": False},
            {"n_items": 10, "max_quantity": 100, "n_breakpoints": 15, "convex": True},
            {"n_items": 10, "max_quantity": 100, "n_breakpoints": 15, "convex": False},
            {"n_items": 30, "max_quantity": 100, "n_breakpoints": 10, "convex": True},
            {"n_items": 30, "max_quantity": 100, "n_breakpoints": 10, "convex": False},
            {"n_items": 50, "max_quantity": 100, "n_breakpoints": 10, "convex": True},
            {"n_items": 50, "max_quantity": 100, "n_breakpoints": 10, "convex": False},
        ],
    },
    "production": {
        "generate": generate_production,
        "solve": solve_production,
        "configs": [
            {
                "n_products": 10,
                "n_resources": 3,
                "max_production": 100,
                "n_breakpoints": 5,
                "convex": True,
            },
            {
                "n_products": 10,
                "n_resources": 3,
                "max_production": 100,
                "n_breakpoints": 5,
                "convex": False,
            },
            {
                "n_products": 10,
                "n_resources": 3,
                "max_production": 100,
                "n_breakpoints": 15,
                "convex": True,
            },
            {
                "n_products": 10,
                "n_resources": 3,
                "max_production": 100,
                "n_breakpoints": 15,
                "convex": False,
            },
            {
                "n_products": 30,
                "n_resources": 5,
                "max_production": 100,
                "n_breakpoints": 10,
                "convex": True,
            },
            {
                "n_products": 30,
                "n_resources": 5,
                "max_production": 100,
                "n_breakpoints": 10,
                "convex": False,
            },
        ],
    },
    "dispatch": {
        "generate": generate_dispatch,
        "solve": solve_dispatch,
        "configs": [
            {"n_generators": 10, "max_output": 500, "n_breakpoints": 5},
            {"n_generators": 10, "max_output": 500, "n_breakpoints": 15},
            {"n_generators": 30, "max_output": 500, "n_breakpoints": 10},
            {"n_generators": 50, "max_output": 500, "n_breakpoints": 10},
        ],
    },
}


def _format_row(seed: int, strategy: str, result: dict) -> str:
    obj_str = f"{result['objective']:.0f}" if result["objective"] is not None else "N/A"
    return (
        f"{seed:>5} | {strategy:>14} | "
        f"{result['status']:>10} {obj_str:>10} "
        f"{result['time_s']:>8.3f} "
        f"{result['num_booleans']:>9} "
        f"{result['num_branches']:>10} "
        f"{result['num_conflicts']:>10}"
    )


def _header() -> str:
    return (
        f"{'seed':>5} | {'strategy':>14} | "
        f"{'status':>10} {'obj':>10} {'time_s':>8} "
        f"{'booleans':>9} {'branches':>10} {'conflicts':>10}"
    )


def run_benchmark(
    problem: str,
    n_seeds: int = 5,
    time_limit: float = 30.0,
    output=None,
) -> None:
    if output is None:
        output = sys.stdout
    spec = PROBLEMS[problem]
    generate = spec["generate"]
    solve = spec["solve"]

    def p(text=""):
        print(text, file=output)

    p(f"\n{'=' * 100}")
    p(f"Problem: {problem}")
    p(f"{'=' * 100}")

    for config in spec["configs"]:
        p(f"\n--- Config: {config} ---")
        p(_header())
        p("-" * 100)

        for seed in range(n_seeds):
            instance = generate(**config, seed=seed)
            for strategy in STRATEGIES:
                result = solve(instance, time_limit, strategy=strategy)
                p(_format_row(seed, strategy, result))


def main():
    parser = argparse.ArgumentParser(description="PWL constraint benchmarks")
    parser.add_argument(
        "--problem",
        choices=list(PROBLEMS.keys()) + ["all"],
        default="all",
        help="Which problem to benchmark (default: all)",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument(
        "--time-limit", type=float, default=30.0, help="Solver time limit (s)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write results to file (in addition to stdout)",
    )
    args = parser.parse_args()

    problems = list(PROBLEMS.keys()) if args.problem == "all" else [args.problem]

    output_file = None
    try:
        if args.output:
            output_file = open(args.output, "w")  # noqa: SIM115

        for problem in problems:
            run_benchmark(
                problem,
                n_seeds=args.seeds,
                time_limit=args.time_limit,
            )
            # Also write to file if requested
            if output_file:
                run_benchmark(
                    problem,
                    n_seeds=args.seeds,
                    time_limit=args.time_limit,
                    output=output_file,
                )
    finally:
        if output_file:
            output_file.close()


if __name__ == "__main__":
    main()
