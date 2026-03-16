# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2026-03-16

### Added

- `PiecewiseLinearFunction.simplify()` public method to remove redundant
  collinear interior breakpoints.
- Pyomo+HiGHS benchmark comparison (`benchmarks/piecewise/compare_pyomo.py`)
  documenting CP-SAT vs MIP solver tradeoffs for piecewise linear problems.

### Fixed

- `__version__` did not match `pyproject.toml` (was still `"0.4.0"`).
- `PiecewiseLinearFunction.__init__` now validates that breakpoint coordinates
  are integers, rejecting floats early with a clear error message.
- `PiecewiseLinearFunction.from_function()` rejects `x_min >= x_max` and
  deduplicates collapsed breakpoints from integer rounding.
- Instance generators in benchmarks now call `.simplify()` to remove collinear
  breakpoints caused by integer rounding during scaling.

## [0.4.0] - 2026-03-15

### Added

- `cpsat_utils.piecewise` module with `PiecewiseLinearFunction` and
  `PiecewiseConstantFunction` for modeling non-linear relationships as integer
  constraints in CP-SAT.
- Upper/lower bound constraints (`add_upper_bound`, `add_lower_bound`) and
  floor/ceil/round equality constraints (`add_floor`, `add_ceil`, `add_round`)
  for piecewise linear functions.
- Ordered-step encoding for piecewise constant (step) function constraints.
- Automatic convex partitioning to minimize boolean selector variables.
- Optional convex envelope as redundant solver hints (requires scipy for
  non-convex functions; fallback algorithm included).
- `name` parameter on all constraint methods to avoid variable name collisions
  when adding multiple piecewise constraints to the same model.
- `PiecewiseLinearFunction.from_function()` constructor for sampling a callable
  at evenly spaced breakpoints.
- `PiecewiseConstantFunction.from_intervals()` constructor from `(start, value)`
  pairs.
- Benchmarks comparing encodings on scheduling and resource-allocation
  instances.
- Three progressive examples with plots: pricing tiers, budget allocation,
  energy dispatch.

### Changed

- `segment_gradients()` now returns exact `fractions.Fraction` values instead of
  floats to avoid precision loss with large coordinates.
- Gradient comparisons in `is_convex()` and `_split_into_convex_parts()` use
  integer cross-products instead of float division.

### Fixed

- Context manager `__exit__` methods (`AssertModelFeasible`,
  `AssertModelInfeasible`, `AssertObjectiveValue`, `AssertOptimalWithinTime`)
  destroyed the original traceback when re-raising exceptions from the `with`
  body.
- Equality constraints on simplified (single-part) functions used the
  unsimplified segment list.
- Round-mode equality constraints now add both upper and lower convex envelopes
  for tighter propagation.
- Internal invariant checks in `_integer_line_coefficients` and convex envelope
  builders now raise `RuntimeError` instead of using `assert`, so they are not
  silently disabled by `python -O`.
- `solve()` docstring incorrectly claimed the `model` parameter accepts objects
  with a `.model` attribute.

## [0.3.0] - 2026-03-15

### Added

- `cpsat_utils.hints` module with `assert_hint_feasible()` and `complete_hint()`
  for validating and completing partial solution hints.
- `cpsat_utils.io` module with `export_model()` and `import_model()` for
  saving/loading models in binary or text protobuf format.
- CI now tests against both ortools 9.10 and the latest version.

### Changed

- README expanded with documentation for hints and io modules.

## [0.2.1] - 2026-03-15

### Fixed

- `solve()` failed with newer ortools versions where `CpSolverStatus` does not
  subclass `int`, causing a `TypeError` on the `status in expect` check.

## [0.2.0] - 2026-03-15

### Added

- `__init__.py` now re-exports the full public API, so
  `from cpsat_utils import assert_feasible` works directly.
- `[project.optional-dependencies]` for `test` and `dev` extras.
- Ruff and pytest configuration in `pyproject.toml`.
- Project URLs (Homepage, Repository, Issues) in `pyproject.toml`.
- Comprehensive test suite covering all context managers and standalone
  functions (26 tests, up from 2).
- README with API documentation and usage examples.

### Fixed

- `AssertObjectiveValue` and `_solve()` used `solver.Solve()` (capital S)
  instead of the modern `solver.solve()` API.
- CI workflow (`pytest.yml`) now uses ruff instead of flake8 and installs the
  package properly via `pip install ".[test]"`.
- Release workflow PyPI URL pointed to `slurminade` instead of `cpsat-utils`.

### Removed

- Empty `setup.py` (unnecessary with `pyproject.toml`).
- Spurious `src/__init__.py` that broke the src-layout.
- Unused `Cython` build dependency.
- Committed `build/` and `egg-info/` directories.

## [0.1.0] - 2025-07-16

### Added

- Initial release with context managers (`AssertModelFeasible`,
  `AssertModelInfeasible`, `AssertObjectiveValue`, `AssertOptimalWithinTime`)
  and standalone assertion functions (`assert_feasible`, `assert_infeasible`,
  `assert_optimal`, `assert_objective`, `solve`).
