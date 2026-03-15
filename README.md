# cpsat-utils

Testing utilities for Google's
[OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver)
solver. Designed for test-driven development of constraint programming models.

For a full walkthrough of test-driven optimization with CP-SAT, see the
[TDD chapter](https://github.com/d-krupke/cpsat-primer) of the CP-SAT Primer.

## Installation

```bash
pip install cpsat-utils
```

## Usage

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
from cpsat_utils.testing import (
    assert_feasible,
    assert_infeasible,
    assert_optimal,
    assert_objective,
)

model = cp_model.CpModel()
x = model.new_bool_var("x")
model.add(x == 1)
model.minimize(x)

# Each function solves the model and checks the result.
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
