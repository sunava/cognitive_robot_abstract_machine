"""ORM-schema profile data classes.

The motion curves/profiles logic was consolidated into
``pycram.robot_plans.actions.composite.tool_based``. This module only keeps the
plain profile data classes that the generated ORM (``ormatic_interface``)
persists, so the ORM stays importable. Remove it once the ORM is regenerated.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ShearProfile:
    depth_max: float
    depth_ramp_end: float
    shear_amp: float
    shear_cycles: float


@dataclass(frozen=True)
class ShearXYProfile:
    shear_amp: float
    shear_cycles: float


@dataclass(frozen=True)
class SpiralProfile:
    r0: float
    r1: float
    cycles: float


@dataclass(frozen=True)
class SweepProfile:
    length: float
    cycles: float
