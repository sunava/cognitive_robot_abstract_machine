"""
Existing physical demonstrations remain valid with the actuated mobile base.
"""

from __future__ import annotations

from cramera.laboratory_physics_program import LaboratoryProgramPhysics
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics

from . import test_laboratory_physics_program as recorded_checks
from . import test_laboratory_pipetting_program as pipetting_checks
from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics


# %% shared physical acceptance criteria
def test_recorded_transfer_keeps_contact_acceptance_with_mobile_base(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    The original recording still needs a measured lift and supported release.
    """
    recorded_checks.TestPhysicalTransfer().test_contact_grasp_lifts_and_releases_into_the_destination(
        mobile_robot
    )


def test_pipetting_keeps_dose_and_contact_acceptance_with_mobile_base(
    mobile_robot: LaboratoryRobotPhysics,
) -> None:
    """
    Both reagent doses retain the original contact and conservation assertions.
    """
    world = LaboratoryProgramPhysics(mobile_robot, mobile_robot.robot_directory)
    pipetting_checks.test_pipetting_contacts_deliver_both_sources_and_return_to_a1(
        world
    )
