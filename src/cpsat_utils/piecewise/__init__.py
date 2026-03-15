"""
Piecewise function constraints for CP-SAT.

Provides piecewise linear and step (piecewise constant) functions that can be
added as constraints to CP-SAT models.

Usage:
    from cpsat_utils.piecewise import (
        PiecewiseLinearFunction,
        StepFunction,
    )
"""

from cpsat_utils.piecewise._constant import (
    StepFunction,
)
from cpsat_utils.piecewise._linear import (
    PiecewiseLinearFunction,
)

__all__ = [
    "PiecewiseLinearFunction",
    "StepFunction",
]
