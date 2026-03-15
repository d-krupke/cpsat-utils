# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
