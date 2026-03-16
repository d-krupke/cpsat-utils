# cpsat-utils

Utilities for Google's
[OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver)
solver. Provides testing helpers, hint management, piecewise linear/constant
function constraints, and model import/export for test-driven development of
constraint programming models.

For a full walkthrough of test-driven optimization with CP-SAT, see the
[TDD chapter](https://github.com/d-krupke/cpsat-primer) of the CP-SAT Primer.

## Installation

```bash
pip install cpsat-utils
```

Supports ortools 9.10 and newer.

## Testing Helpers

### Context Managers

Assert feasibility or infeasibility of a model built inside a `with` block. The
model is solved automatically when the block exits.

```python
from cpsat_utils.testing import AssertModelFeasible, AssertModelInfeasible

def test_feasible():
    with AssertModelFeasible() as model:
        x = model.new_bool_var("x")
        y = model.new_int_var(0, 10, "y")
        model.add(x + y == 1)

def test_infeasible():
    with AssertModelInfeasible() as model:
        x = model.new_bool_var("x")
        y = model.new_bool_var("y")
        model.add(x + y == 3)
```

Assert that the optimal objective matches an expected value:

```python
from cpsat_utils.testing import AssertObjectiveValue

def test_objective():
    with AssertObjectiveValue(objective=1.0) as model:
        x = model.new_bool_var("x")
        y = model.new_bool_var("y")
        model.add(x + y >= 1)
        model.minimize(x + y)
```

Assert optimality within a time limit:

```python
from cpsat_utils.testing import AssertOptimalWithinTime

def test_optimal():
    with AssertOptimalWithinTime(time_limit=2.0) as model:
        x = model.new_bool_var("x")
        model.minimize(x)
```

### Standalone Functions

For cases where you build the model separately (e.g., testing individual modules
of a larger model):

```python
from ortools.sat.python import cp_model
from cpsat_utils.testing import assert_feasible, assert_optimal, assert_objective

model = cp_model.CpModel()
x = model.new_bool_var("x")
model.add(x == 1)
model.minimize(x)

assert_feasible(model)
assert_optimal(model)

# Check objective value (solves internally).
solver = assert_objective(model, expected=1.0)
# The returned solver can be inspected further.
assert solver.value(x) == 1
```

You can also pass an explicit solver to inspect variable values afterward:

```python
solver = cp_model.CpSolver()
assert_objective(model=model, solver=solver, expected=1.0)
assert solver.value(x) == 1
```

## Hint Utilities

### Validating Hints

Check that hints are feasible before committing to a long solve:

```python
from cpsat_utils.hints import assert_hint_feasible

model.add_hint(x, 1)
model.add_hint(y, 0)
assert_hint_feasible(model)  # raises if hints are infeasible
```

### Completing Partial Hints

CP-SAT benefits most from complete hints (all variables hinted). If you only
have values for some variables, `complete_hint` fills in the rest via a quick
solve:

```python
from cpsat_utils.hints import complete_hint

model.add_hint(x, 1)  # only hint x
complete_hint(model)   # fills in y, z, ... via a short solve
```

Returns `True` on success, `False` if the hints are infeasible or the solve
times out (hints are left unchanged on failure).

## Piecewise Linear Functions

Model non-linear relationships (costs, revenue, value curves) as integer
constraints in CP-SAT. This is useful when piecewise functions appear as part
of a larger model that benefits from CP-SAT's strengths in combinatorial
optimization. For pure non-linear optimization, dedicated solvers are
typically a better choice.

### Why not just `y = f(x)`?

CP-SAT works with integers only, which creates two problems for piecewise
linear functions (see the
[CP-SAT Primer](https://d-krupke.github.io/cpsat-primer/04B_advanced_modelling.html#non-linear-constraintspiecewise-linear-functions)
for an in-depth explanation):

1. **Non-integral values:** For most integer `x`, `f(x)` falls between
   integers (e.g., `f(5) = 3.5`), making `y = f(x)` infeasible. That is why
   the API offers one-sided bounds (`add_upper_bound`, `add_lower_bound`) and
   rounding modes (`add_floor`, `add_ceil`, `add_round`) instead of plain
   equality.
2. **Non-integral coefficients:** The slope `a` and intercept `b` in
   `y = ax + b` are often fractional. Internally, each segment is scaled to
   integer form `t*y = a*x + b` using the least common multiple. When `dy`
   and `dx` of a segment are large coprimes, the resulting coefficients can
   become very large — the constructor warns when `lcm(|dy|, dx) > 10^9`.

### Input validation

All breakpoints must be integers (both `xs` and `ys`). Passing floats raises
a `TypeError`. When using `from_function`, y-values are automatically
rounded and duplicate x-values (from rounding in small ranges) are
deduplicated with a warning.

```python
from ortools.sat.python import cp_model
from cpsat_utils.piecewise import PiecewiseLinearFunction

model = cp_model.CpModel()
x = model.new_int_var(0, 100, "x")

# Define from breakpoints:
f = PiecewiseLinearFunction([0, 30, 70, 100], [0, 80, 60, 100])

# One-sided bounds — the optimizer pushes y to the bound:
y = f.add_upper_bound(model, x)  # y <= f(x), use when maximizing y
y = f.add_lower_bound(model, x)  # y >= f(x), use when minimizing y

# Equality constraints — exact integer rounding:
y = f.add_floor(model, x)   # y = floor(f(x))
y = f.add_ceil(model, x)    # y = ceil(f(x))
y = f.add_round(model, x)   # y = round(f(x))
```

Approximate any callable as a piecewise linear function:

```python
import math
f = PiecewiseLinearFunction.from_function(math.sqrt, x_min=0, x_max=100, num_breakpoints=20)
y = f.add_round(model, x)   # y ≈ sqrt(x)
```

### Encoding optimizations

The implementation automatically applies two optimizations that dramatically
improve solver performance:

- **Convex partitioning** (one-sided bounds only): groups consecutive segments
  with compatible gradients into convex parts, reducing the number of boolean
  selector variables. A function with 50 segments but only 3 convex parts
  needs 3 booleans instead of 50.
- **Convex envelope**: adds a redundant global constraint that tightens the
  LP relaxation without reification. This is the dominant optimization —
  it enables the solver to prove optimality with zero branching on many
  instances.

Both are enabled by default. On benchmarks with 5000 piecewise linear
functions (10 breakpoints each), `bound+opt+env` solves a knapsack in 15s
and a generator dispatch in 19s. Without the envelope, instances with 50
functions already time out at 30s. See
[`benchmarks/piecewise/README.md`](benchmarks/piecewise/README.md) for full
results.

### Step functions

For piecewise constant functions (e.g., pricing tiers, tax brackets):

```python
from cpsat_utils.piecewise import StepFunction

# Value is 10 for x in [0,3), 20 for x in [3,7), 30 for x in [7,10)
f = StepFunction([0, 3, 7, 10], [10, 20, 30])
y = f.add_constraint(model, x)
```

### Examples

See [`examples/`](examples/README.md) for complete, runnable examples with
plots — from a simple budget allocation (PiecewiseLinearFunction) to
multi-period generator dispatch (both function types combined).

## Model Import/Export

Save and load models for comparing solver performance across machines or ortools
versions, without sharing code:

```python
from cpsat_utils.io import export_model, import_model

# Export (format detected by extension)
export_model(model, "my_model.pb")      # binary protobuf
export_model(model, "my_model.pbtxt")   # human-readable text

# Import
loaded = import_model("my_model.pb")
```

Supported extensions: `.pb`, `.bin`, `.dat` (binary) and `.txt`, `.pbtxt`
(text).
