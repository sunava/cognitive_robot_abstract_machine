"""
Smooth conical container motion about a stationary grip or rim pivot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.spatial.transform import Rotation

# %% Motion parameters


@dataclass
class InvalidSwirl(ValueError):
    """
    Swirling parameters do not define a finite, rigid container trajectory.
    """

    reason: str
    """The violated trajectory requirement."""


@dataclass(frozen=True)
class SwirlProfile:
    """
    A conical rotation with smooth entry and exit and no axial spin.
    """

    tilt_angle: float = 0.20
    """
    Largest angle from the initial container axis, in radians.
    """

    cycles: float = 3.0
    """
    Number of azimuth revolutions, including the entry and exit ramps.
    """

    duration: float = 8.0
    """
    Total trajectory duration in seconds, including both ramps.
    """

    ramp_fraction: float = 0.15
    """
    Fraction of the duration spent smoothly entering and leaving the cone.
    """

    def __post_init__(self) -> None:
        """
        Reject nonfinite parameters, inverted containers, and empty motions.
        """
        if not all(
            math.isfinite(value)
            for value in (
                self.tilt_angle,
                self.cycles,
                self.duration,
                self.ramp_fraction,
            )
        ):
            raise InvalidSwirl("Motion parameters must be finite.")
        if not 0 < self.tilt_angle < math.pi / 2:
            raise InvalidSwirl("Tilt must lie between zero and a right angle.")
        if self.cycles <= 0 or self.duration <= 0:
            raise InvalidSwirl("Cycles and duration must be positive.")
        if not 0 < self.ramp_fraction <= 0.5:
            raise InvalidSwirl("Each ramp must occupy at most half the motion.")

    @classmethod
    def from_radius(
        cls,
        bottom_radius: float,
        pivot_to_bottom: float,
        *,
        cycles: float = 3.0,
        duration: float = 8.0,
        ramp_fraction: float = 0.15,
    ) -> SwirlProfile:
        """
        Construct the tilt that traces a requested bottom-circle radius.

        :param bottom_radius: Radius in meters at full tilt.
        :param pivot_to_bottom: Distance along the container axis from pivot to bottom.
        :param cycles: Number of azimuth revolutions.
        :param duration: Total motion duration in seconds.
        :param ramp_fraction: Fraction of duration allocated to each ramp.
        :return: Profile whose peak bottom orbit has the requested radius.
        """
        if (
            not math.isfinite(pivot_to_bottom)
            or not 0 < bottom_radius < pivot_to_bottom
        ):
            raise InvalidSwirl(
                "Bottom radius must be smaller than the positive pivot distance."
            )
        return cls(
            math.asin(bottom_radius / pivot_to_bottom), cycles, duration, ramp_fraction
        )

    def rotation_at(self, elapsed: float) -> np.ndarray:
        """
        Return the rotation in the initial container frame at elapsed seconds.

        The container's local positive Z axis is its long axis. Its lower end orbits in
        local XY, while the shortest rotation to each tilted axis avoids axial spin.
        """
        if not math.isfinite(elapsed):
            raise InvalidSwirl("Elapsed time must be finite.")
        progress = float(np.clip(elapsed / self.duration, 0.0, 1.0))
        ramp_progress = (
            min(progress, 1.0 - progress, self.ramp_fraction) / self.ramp_fraction
        )
        envelope = ramp_progress**3 * (
            10.0 - 15.0 * ramp_progress + 6.0 * ramp_progress**2
        )
        phase = 2.0 * math.pi * self.cycles * progress
        tilt_axis = np.array([math.sin(phase), -math.cos(phase), 0.0])
        return Rotation.from_rotvec(self.tilt_angle * envelope * tilt_axis).as_matrix()


# %% Container pose trajectory


@dataclass(frozen=True)
class SwirlTrajectory:
    """
    Rigid container poses that preserve a pivot throughout a conical swirl.
    """

    root_T_container: np.ndarray
    """
    Initial homogeneous container transform; its local XY plane defines the orbit.
    """

    container_P_pivot: np.ndarray
    """
    Stationary pivot expressed as three coordinates in the container frame.
    """

    profile: SwirlProfile = field(default_factory=SwirlProfile)
    """
    Timing and tilt of the conical motion.
    """

    def __post_init__(self) -> None:
        """
        Validate and copy the rigid transform and local pivot.
        """
        transform = np.array(self.root_T_container, dtype=float, copy=True)
        pivot = np.array(self.container_P_pivot, dtype=float, copy=True)
        if (
            transform.shape != (4, 4)
            or pivot.shape != (3,)
            or not np.all(np.isfinite(transform))
            or not np.all(np.isfinite(pivot))
        ):
            raise InvalidSwirl(
                "A finite 4 by 4 transform and three-coordinate pivot are required."
            )
        rotation = transform[:3, :3]
        if (
            not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(rotation.T @ rotation, np.eye(3))
            or not np.isclose(np.linalg.det(rotation), 1.0)
        ):
            raise InvalidSwirl(
                "The initial transform must contain a proper rigid rotation."
            )
        transform.setflags(write=False)
        pivot.setflags(write=False)
        object.__setattr__(self, "root_T_container", transform)
        object.__setattr__(self, "container_P_pivot", pivot)

    def pose_at(self, elapsed: float) -> np.ndarray:
        """
        Return the container pose while keeping its pivot fixed in the root frame.
        """
        container_R_tilted = self.profile.rotation_at(elapsed)
        container_T_tilted = np.eye(4)
        container_T_tilted[:3, :3] = container_R_tilted
        container_T_tilted[:3, 3] = (
            self.container_P_pivot - container_R_tilted @ self.container_P_pivot
        )
        return self.root_T_container @ container_T_tilted
