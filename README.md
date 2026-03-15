# cpsat-utils

Utilities for Google's
[OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver)
solver. Provides testing helpers, hint management, and model import/export for
test-driven development of constraint programming models.

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
