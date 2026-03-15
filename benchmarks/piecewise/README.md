# Piecewise Linear Constraint Benchmarks

Benchmarks comparing different encoding strategies for piecewise linear (PWL)
constraints in CP-SAT. The goal is to measure how convex partitioning and
convex envelope redundant constraints affect solver performance.

## Running

```bash
python -m benchmarks.piecewise                          # all problems, 5 seeds, 30s limit
python -m benchmarks.piecewise --problem knapsack       # single problem
python -m benchmarks.piecewise --seeds 3 --time-limit 10
```

## Problems

All three problems use piecewise linear functions to model non-linear
relationships (value, revenue, cost) that arise naturally in operations
research. Each problem couples multiple PWL constraints through shared
resource or demand constraints, making the encoding quality visible in
solver performance.

### 1. Piecewise Knapsack

**Setting.** A set of items, each with a quantity variable `q_i` and a
piecewise linear value curve `v_i(q_i)` describing the total value obtained
from selecting `q_i` units. Each item has a per-unit weight `w_i`. All items
share a single knapsack capacity.

**Instance.**

- `n_items` items, each with:
  - A PWL value curve `value_i: [0, max_quantity] -> [0, ~1000]`
    (concave for `convex=True`, random shape for `convex=False`)
  - A per-unit weight `w_i in [1, 10]`
- Shared capacity = `0.4 * n_items * max_quantity`

**Model.**

```
Variables:  q_i in [0, max_quantity]  for each item i
            v_i                       value from item i (PWL-constrained)

Maximize:   sum(v_i)
Subject to: v_i <= value_i(q_i)      (or v_i = floor(value_i(q_i)) for equality)
            sum(q_i * w_i) <= capacity
```

**Solution.** An assignment of quantities to items that maximizes total value
without exceeding the knapsack capacity. The concave (diminishing returns)
curves make this a non-trivial allocation problem: spreading quantity across
items captures diminishing-returns benefits, but the shared capacity forces
trade-offs.

### 2. Multi-Product Production Planning

**Setting.** A factory produces multiple products. Each product has a
piecewise linear revenue curve describing how revenue depends on production
quantity (e.g., volume discounts, market saturation). Products compete for
shared resources (machines, labor, raw materials).

**Instance.**

- `n_products` products, each with:
  - A PWL revenue curve `revenue_i: [0, max_production] -> [0, ~10000]`
  - Resource usage `usage_i[r]` per unit, for each resource `r`
    (`usage in [1, 5]`)
- `n_resources` shared resources, each with capacity =
  `0.6 * sum(usage_i[r] * max_production)` (tight enough to force trade-offs)

**Model.**

```
Variables:  q_i in [0, max_production]  for each product i
            rev_i                       revenue from product i (PWL-constrained)

Maximize:   sum(rev_i)
Subject to: rev_i <= revenue_i(q_i)
            sum(q_i * usage_i[r]) <= capacity[r]   for each resource r
```

**Solution.** A production plan allocating capacity across products to
maximize total revenue. The multiple shared resources create a richer
coupling than the single-constraint knapsack.

### 3. Generator Dispatch (Unit Commitment)

**Setting.** A set of power generators must collectively meet an electricity
demand target. Each generator has a piecewise linear cost curve describing
how fuel cost depends on power output (increasing marginal cost due to
efficiency losses at high output). Generators can be switched on or off;
when on, they must operate within their output range.

**Instance.**

- `n_generators` generators, each with:
  - A PWL cost curve `cost_i: [p_min_i, p_max_i] -> [0, ~5000]`
    (convex increasing — increasing marginal cost)
  - Minimum output `p_min_i in [0.1*max, 0.3*max]`
  - Maximum output `p_max_i in [0.7*max, max]`
- Demand target = `0.6 * sum(p_max_i)`

**Model.**

```
Variables:  on_i in {0, 1}              whether generator i is on
            p_i in [0, p_max_i]         power output of generator i
            c_i                         cost of generator i (PWL-constrained)

Minimize:   sum(c_i)
Subject to: c_i >= cost_i(p_i)          (or c_i = ceil(cost_i(p_i)) for equality)
            p_i >= p_min_i  if on_i     (minimum output when on)
            p_i <= p_max_i  if on_i
            p_i == 0        if !on_i    (zero output when off)
            sum(p_i) >= demand
```

**Solution.** A dispatch plan selecting which generators to activate and at
what output level, minimizing total fuel cost while meeting demand. The
on/off decisions combined with convex cost curves make this the most
structurally complex of the three problems.

## Encoding Strategies

Each PWL function `f(x)` must be encoded as CP-SAT constraints. Since CP-SAT
operates on integers, `f(x)` generally falls between integers and cannot be
enforced as an exact equality. The benchmark compares five encoding strategies:

| Strategy | Constraint type | Partition | Envelope | Description |
|---|---|---|---|---|
| `bound+opt+env` | one-sided bound | convex | yes | Best formulation: groups segments into convex parts (fewer booleans) + adds convex hull as redundant global constraint |
| `bound+opt` | one-sided bound | convex | no | Convex partitioning only, no envelope |
| `bound+naive` | one-sided bound | per-segment | no | Baseline: one boolean selector per segment, no optimizations |
| `eq+env` | equality (floor/ceil) | per-segment | yes | Two-sided constraints per segment + convex envelope |
| `eq+naive` | equality (floor/ceil) | per-segment | no | Two-sided constraints per segment, no optimizations |

**One-sided bound** (`add_upper_bound` / `add_lower_bound`): constrains
`y <= f(x)` or `y >= f(x)`. The optimizer pushes `y` to the bound. Within a
convex region, all segment constraints are simultaneously satisfiable, so
convex partitioning reduces the number of boolean selector variables.

**Equality** (`add_floor` / `add_ceil`): constrains `y = floor(f(x))` or
`y = ceil(f(x))` using two-sided per-segment constraints. The tightened side
from non-active segments conflicts, so convex partitioning cannot reduce
booleans here. The convex envelope still helps as a global redundant
constraint.

**Convex envelope**: a redundant constraint that bounds `y` globally without
reification. For upper bounds, it adds `y <= hull(x)` where `hull` is the
tightest convex function above `f`. This helps the solver's LP relaxation
produce tighter bounds, dramatically reducing the search space.

## Instance Configurations

### Knapsack

| n_items | max_quantity | n_breakpoints | convex |
|---|---|---|---|
| 10 | 100 | 5 | True / False |
| 10 | 100 | 15 | True / False |
| 30 | 100 | 10 | True / False |
| 50 | 100 | 10 | True / False |

### Production

| n_products | n_resources | max_production | n_breakpoints | convex |
|---|---|---|---|---|
| 10 | 3 | 100 | 5 | True / False |
| 10 | 3 | 100 | 15 | True / False |
| 30 | 5 | 100 | 10 | True / False |

### Dispatch

| n_generators | max_output | n_breakpoints |
|---|---|---|
| 10 | 500 | 5 |
| 10 | 500 | 15 |
| 30 | 500 | 10 |
| 50 | 500 | 10 |

## Key Results (2026-03-15, 10s time limit, 3 seeds)

The convex envelope is the dominant optimization. Without it, larger instances
cannot be solved to optimality within the time limit.

| Problem | Config | `bound+opt+env` | `eq+env` | `bound+opt` | `bound+naive` |
|---|---|---|---|---|---|
| Knapsack | 50 items, convex | 0.02s OPTIMAL | 0.05s OPTIMAL | 2-10s+ | 10s+ FEASIBLE |
| Knapsack | 50 items, non-convex | 0.05s OPTIMAL | 0.09s OPTIMAL | 10s+ FEASIBLE | 10s+ FEASIBLE |
| Production | 30 prod, convex | 0.2s OPTIMAL | 0.4s OPTIMAL | 3-5s | 10s+ FEASIBLE |
| Production | 30 prod, non-convex | 0.3s OPTIMAL | 0.5s OPTIMAL | 10s+ FEASIBLE | 10s+ FEASIBLE |
| Dispatch | 50 gen | 0.015s OPTIMAL | 0.05s OPTIMAL | 1-3s | 10s+ FEASIBLE |

**Observations:**

- `bound+opt+env` is consistently the fastest strategy (fewest booleans +
  envelope).
- `eq+env` is a strong second: despite more boolean variables (per-segment
  rather than per-convex-part), the envelope compensates. It always finds the
  optimal solution within 1 second on these instances.
- Without the envelope (`bound+opt`, `eq+naive`, `bound+naive`), the solver
  struggles to close the optimality gap on larger instances, often hitting
  the time limit with only a feasible (not proven optimal) solution.
- The envelope alone accounts for the majority of the speedup. Convex
  partitioning provides an additional but smaller benefit on top.

## Scaling Limits (2026-03-15, 30s time limit, seed 0)

Stress testing the two best strategies (`bound+opt+env` and `eq+env`) on
increasingly large instances reveals where each problem becomes hard.

### Knapsack — scales very well

The single shared capacity constraint makes the LP relaxation easy. Both
strategies solve massive instances within the time limit.

| Config | `bound+opt+env` | `eq+env` |
|---|---|---|
| 500 items, 10bp, convex | 0.3s | 0.7s |
| 1000 items, 10bp, convex | 1.0s | 2.4s |
| 2000 items, 10bp, convex | 2.2s | 4.4s |
| 5000 items, 10bp, convex | 15s | 21s |
| 1000 items, 10bp, non-convex | 3.3s | 3.0s |
| 200 items, 90bp, convex | 0.7s | 2.1s |
| 200 items, 90bp, non-convex | 1.2s | 3.7s |

### Production — the bottleneck

Multiple shared resource constraints couple all product variables, making the
LP relaxation much harder. The number of resources is the dominant difficulty
driver, not the number of products or breakpoints.

| Config | `bound+opt+env` | `eq+env` |
|---|---|---|
| 100 prod, 5 res, convex | 0.7s | 1.3s |
| 200 prod, 5 res, convex | 15s | 22s |
| 300 prod, 5 res, convex | 15s | 18s |
| 100 prod, 5 res, non-convex | 5s | 14s |
| 150 prod, 5 res, non-convex | 23s | **30s+ FEASIBLE** |
| 200 prod, 5 res, non-convex | **30s+ FEASIBLE** | **30s+ FEASIBLE** |
| 50 prod, 10 res, convex | 1.8s | 2.0s |
| 50 prod, 20 res, convex | **30s+ FEASIBLE** | **30s+ FEASIBLE** |
| 50 prod, 10 res, non-convex | 1.6s | 3.9s |

### Dispatch — scales well

The on/off binary decisions add some branching, but the single demand
constraint keeps the LP relaxation tight.

| Config | `bound+opt+env` | `eq+env` |
|---|---|---|
| 500 gen, 10bp | 0.3s | 0.7s |
| 1000 gen, 10bp | 0.8s | 2.0s |
| 2000 gen, 10bp | 5.0s | 6.4s |
| 5000 gen, 10bp | 19s | 28s |
| 200 gen, 90bp | 1.2s | 3.7s |
| 500 gen, 50bp | 1.4s | 4.8s |

### Key takeaways

- **Scaling driver is constraint coupling, not PWL complexity.** Problems with
  a single shared constraint (knapsack, dispatch) handle 5000 PWL functions.
  Production with 20+ resource constraints times out at just 50 products.
- **`bound+opt+env` is consistently faster** — roughly 2-3x fewer booleans
  and 1.5-2x faster solve times across all problems.
- **Both strategies handle thousands of PWL functions** when the coupling
  structure is simple. The convex envelope makes the LP relaxation so tight
  that the solver often needs zero branches.

## File Structure

```
benchmarks/piecewise/
    __init__.py
    __main__.py        CLI entry point and problem configurations
    _instances.py      Dataclasses and random instance generators
    _solvers.py        Solve functions for each problem type
    README.md          This file
    _stress.py         Scaling stress test for best strategies
    results/
        2026-03-15.txt         Full benchmark output (5 strategies)
        2026-03-15-stress.txt  Stress test output (scaling limits)
```
