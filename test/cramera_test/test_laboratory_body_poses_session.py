"""
Initial viewer frames preserve measured drawer and independent carrier poses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from cramera import laboratory_physics_session, laboratory_robot_session
from cramera.laboratory_drawer import DrawerField, DrawerGeometry
from cramera.laboratory_physics_session import (
    LaboratoryPhysicsSession,
    PhysicsField,
    PhysicsFile,
    PhysicsState,
)
from cramera.laboratory_rack import RackField, RackGeometry
from cramera.laboratory_robot_session import LaboratoryRobotSession

from .test_laboratory_physics_session import ControlledContactSimulation, physics_bundle
from .test_laboratory_robot_session import ControlledRobotSimulation, robot_bundle


# %% measured initial placement
@dataclass
class CarriedRackObservation:
    """
    Supply observed carrier motion independently of robot joints and samples.
    """

    @staticmethod
    def snapshot() -> dict:
        """
        Describe a partially opened drawer with a lifted rack.
        """
        return {
            RackField.BODY_POSES: {RackGeometry.BODY: [-0.52, -0.4, 0.8, 0, 0, 0, 1]},
            DrawerField.DRAWER: {DrawerField.JOINTS: {DrawerGeometry.JOINT: 0.27}},
        }


@dataclass
class CarriedRackContactSimulation(ControlledContactSimulation):
    """
    Expose free-carrier and drawer observations with measured sample positions.
    """

    def snapshot(self) -> dict:
        """
        Combine the independent body and articulated drawer observations.
        """
        return {**super().snapshot(), **CarriedRackObservation.snapshot()}


@dataclass
class CarriedRackRobotSimulation(ControlledRobotSimulation):
    """
    Expose free-carrier observations alongside physical robot joints.
    """

    def snapshot(self) -> dict:
        """
        Retain the carrier and drawer independently of the robot joint map.
        """
        return {**super().snapshot(), **CarriedRackObservation.snapshot()}


@pytest.mark.parametrize("robot", [False, True])
def test_initial_trajectory_retains_measured_drawer_and_rack(
    physics_bundle, robot_bundle, monkeypatch, robot: bool
) -> None:
    """
    Reloading either scene starts at the measured carrier and drawer position.
    """
    monkeypatch.setattr(
        laboratory_physics_session, "create_physics", CarriedRackContactSimulation
    )
    monkeypatch.setattr(
        laboratory_robot_session, "create_robot_physics", CarriedRackRobotSimulation
    )
    session = (LaboratoryRobotSession if robot else LaboratoryPhysicsSession)(
        physics_bundle.parents[1]
    )
    try:
        result = session.start()
        assert result[PhysicsField.STATE] == PhysicsState.RUNNING
        trajectory = json.loads(
            (session.output_directory / PhysicsFile.TRAJECTORY).read_text()
        )
        observed = CarriedRackObservation.snapshot()
        assert trajectory[RackField.BODY_POSES] == [observed[RackField.BODY_POSES]]
        assert (
            trajectory[PhysicsField.FRAMES][0][DrawerGeometry.JOINT]
            == observed[DrawerField.DRAWER][DrawerField.JOINTS][DrawerGeometry.JOINT]
        )
    finally:
        session.stop()
