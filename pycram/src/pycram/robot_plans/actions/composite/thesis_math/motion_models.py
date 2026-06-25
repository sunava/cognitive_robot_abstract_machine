"""ORM-schema data classes for sampled pose sequences.

The motion logic (MotionSegment / MotionSequence and sampling) was consolidated
into ``pycram.robot_plans.actions.composite.tool_based``. This module only keeps
the plain data class that the generated ORM (``ormatic_interface``) persists, so
the ORM stays importable. Remove it once the ORM is regenerated.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SampledPoseSequence:
    times: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray
    phase_ids: np.ndarray
