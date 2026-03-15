"""
Benchmark and stress test for StepFunction constraints.

Tests the current ordered-step encoding on knapsack-style problems
with step-function values. Compares against a table-constraint baseline
to identify potential improvements.

Usage:
    python -m benchmarks.piecewise._stress_constant

When to modify:
    - To add alternative encodings for comparison
    - To adjust instance sizes
"""

from __future__ import annotations

import random
import time

from ortools.sat.python import cp_model

from cpsat_utils.piecewise import StepFunction

TIME_LIMIT = 30.0


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------


def _random_step_function(
    x_max: int, n_pieces: int, y_scale: int, rng: random.Random
) -> StepFunction:
    """Random step function with n_pieces intervals over [0, x_max)."""
    xs = sorted(rng.sample(range(1, x_max), n_pieces - 1))
    xs = [0, *xs, x_max]
    ys = [rng.randint(0, y_scale) for _ in range(n_pieces)]
    return StepFunction(xs, ys)


def _random_monotone_step_function(
    x_max: int, n_pieces: int, y_scale: int, rng: random.Random
) -> StepFunction:
    """Random non-decreasing step function."""
    xs = sorted(rng.sample(range(1, x_max), n_pieces - 1))
    xs = [0, *xs, x_max]
    ys = sorted(rng.randint(0, y_scale) for _ in range(n_pieces))
    return StepFunction(xs, ys)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


def solve_step_knapsack(
    curves: list[StepFunction],
    weights: list[int],
    capacity: int,
    time_limit: float,
    encoding: str = "step",
) -> dict:
    """Knapsack with step-function values."""
    model = cp_model.CpModel()
    quantities = []
    total_value = 0

    for i, (curve, _w) in enumerate(zip(curves, weights, strict=True)):
        q = model.new_int_var(0, curve.x_max, f"q_{i}")
        quantities.append(q)

        if encoding == "step":
            v = curve.add_constraint(model, q, name=f"v_{i}")
        elif encoding == "table":
            v = _add_table_constraint(model, curve, q, f"v_{i}")
        elif encoding == "element":
            v = _add_element_constraint(model, curve, q, f"v_{i}")
        else:
            msg = f"Unknown encoding: {encoding}"
            raise ValueError(msg)
        total_value += v

    model.add(sum(q * w for q, w in zip(quantities, weights, strict=True)) <= capacity)
    model.maximize(total_value)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    t0 = time.perf_counter()
    status = solver.solve(model)
    elapsed = time.perf_counter() - t0
    is_feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    return {
        "status": solver.status_name(status),
        "objective": solver.objective_value if is_feasible else None,
        "time_s": round(elapsed, 3),
        "num_booleans": solver.num_booleans,
        "num_branches": solver.num_branches,
    }


def _add_table_constraint(
    model: cp_model.CpModel,
    f: StepFunction,
    x: cp_model.IntVar,
    name: str,
) -> cp_model.IntVar:
    """Encode step function via AddAllowedAssignments (table constraint)."""
    y = model.new_int_var(min(f.ys), max(f.ys), name)
    model.add(x >= f.xs[0])
    model.add(x <= f.xs[-1] - 1)
    tuples = []
    for i in range(len(f.ys)):
        for xv in range(f.xs[i], f.xs[i + 1]):
            tuples.append((xv, f.ys[i]))
    model.add_allowed_assignments([x, y], tuples)
    return y


def _add_element_constraint(
    model: cp_model.CpModel,
    f: StepFunction,
    x: cp_model.IntVar,
    name: str,
) -> cp_model.IntVar:
    """Encode step function via AddElement (lookup table)."""
    y = model.new_int_var(min(f.ys), max(f.ys), name)
    model.add(x >= f.xs[0])
    model.add(x <= f.xs[-1] - 1)
    # Build a flat lookup: value_at[x] = f(x) for each integer x
    lookup = []
    for i in range(len(f.ys)):
        lookup.extend([f.ys[i]] * (f.xs[i + 1] - f.xs[i]))
    # AddElement: lookup[x - offset] == y
    offset = f.xs[0]
    idx = model.new_int_var(0, len(lookup) - 1, f"{name}_idx")
    model.add(idx == x - offset)
    model.add_element(idx, lookup, y)
    return y


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


ENCODINGS = ["step", "table", "element"]


def _run(label: str, configs: list[dict]) -> None:
    print(f"\n{'=' * 95}")
    print(f"Stress test: {label}")
    print(f"{'=' * 95}")
    print(
        f"{'config':<40} | {'encoding':>8} | "
        f"{'status':>10} {'time_s':>8} {'bools':>7} {'branches':>10}"
    )
    print("-" * 95)

    for config in configs:
        n_items = config["n_items"]
        x_max = config["x_max"]
        n_pieces = config["n_pieces"]
        monotone = config.get("monotone", False)

        config_str = f"n={n_items}, xmax={x_max}, pieces={n_pieces}"
        if monotone:
            config_str += ", mono"
        if len(config_str) > 38:
            config_str = config_str[:35] + "..."

        rng = random.Random(42)
        gen = _random_monotone_step_function if monotone else _random_step_function
        curves = [gen(x_max, n_pieces, 1000, rng) for _ in range(n_items)]
        weights = [rng.randint(1, 10) for _ in range(n_items)]
        capacity = int(0.4 * n_items * x_max)

        for enc in ENCODINGS:
            # Skip table encoding for large domains (would create huge tuples)
            if enc == "table" and n_items * x_max > 50000:
                print(
                    f"{config_str:<40} | {enc:>8} | "
                    f"{'SKIPPED':>10} {'':>8} {'':>7} {'':>10}"
                )
                continue
            result = solve_step_knapsack(
                curves, weights, capacity, TIME_LIMIT, encoding=enc
            )
            status = result["status"]
            marker = " ***" if status != "OPTIMAL" else ""
            print(
                f"{config_str:<40} | {enc:>8} | "
                f"{status:>10} {result['time_s']:>8.3f} "
                f"{result['num_booleans']:>7} {result['num_branches']:>10}"
                f"{marker}"
            )
        print()


def main():
    _run(
        "Step-function knapsack — scaling items",
        [
            {"n_items": 10, "x_max": 100, "n_pieces": 5},
            {"n_items": 50, "x_max": 100, "n_pieces": 5},
            {"n_items": 100, "x_max": 100, "n_pieces": 5},
            {"n_items": 200, "x_max": 100, "n_pieces": 5},
            {"n_items": 500, "x_max": 100, "n_pieces": 5},
        ],
    )

    _run(
        "Step-function knapsack — scaling pieces",
        [
            {"n_items": 50, "x_max": 100, "n_pieces": 5},
            {"n_items": 50, "x_max": 100, "n_pieces": 10},
            {"n_items": 50, "x_max": 100, "n_pieces": 20},
            {"n_items": 50, "x_max": 100, "n_pieces": 50},
            {"n_items": 50, "x_max": 100, "n_pieces": 90},
        ],
    )

    _run(
        "Step-function knapsack — scaling x domain",
        [
            {"n_items": 50, "x_max": 100, "n_pieces": 10},
            {"n_items": 50, "x_max": 500, "n_pieces": 10},
            {"n_items": 50, "x_max": 1000, "n_pieces": 10},
            {"n_items": 50, "x_max": 5000, "n_pieces": 10},
        ],
    )

    _run(
        "Step-function knapsack — monotone functions",
        [
            {"n_items": 50, "x_max": 100, "n_pieces": 10, "monotone": True},
            {"n_items": 100, "x_max": 100, "n_pieces": 10, "monotone": True},
            {"n_items": 200, "x_max": 100, "n_pieces": 10, "monotone": True},
            {"n_items": 50, "x_max": 100, "n_pieces": 50, "monotone": True},
        ],
    )


if __name__ == "__main__":
    main()
