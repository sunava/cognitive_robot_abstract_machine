"""
Native semantic plans observe the physically moving rack and separate samples.
"""

from pathlib import Path

import numpy as np

from cramera.laboratory_handles import LaboratoryHandle
from cramera.laboratory_physics_robot import LaboratoryRobotPhysics
from cramera.laboratory_plan_world import LaboratoryPlanWorld
from cramera.laboratory_rack import RackGeometry
from cramera.laboratory_world import LaboratoryWorld
from semantic_digital_twin.world_description.connections import Connection6DoF

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_world import laboratory_bundle


# %% detached moving rack measurements
def test_native_rack_handle_and_samples_follow_captured_drawer_motion(
    mobile_robot: LaboratoryRobotPhysics, laboratory_bundle: Path
) -> None:
    """
    Applying a captured pose moves the rack frame and handle without moving samples
    twice.
    """
    physics = mobile_robot
    world = LaboratoryWorld(bundle_directory=laboratory_bundle).build()
    rack = world.get_body_by_name(RackGeometry.BODY)
    assert isinstance(rack.parent_connection, Connection6DoF)
    physics.prepare_rack_in_drawer()
    physics.step(500)
    physics.drawer.hold(physics.drawer.open_position)
    physics.step(700)
    measurement = LaboratoryPlanWorld.capture(physics)
    captured = {pose.body_name: pose for pose in measurement.poses}
    handle = physics.model.site(LaboratoryHandle.RACK).id
    handle_position = physics.data.site_xpos[handle].copy()
    physics.step(100)
    before = physics.data.qpos.copy()
    measurement.apply(world)
    for name in (RackGeometry.BODY, *physics.liquid.tubes):
        np.testing.assert_allclose(
            world.get_body_by_name(name).global_pose.to_np()[:3, 3],
            captured[name].position,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        world.get_body_by_name(LaboratoryHandle.RACK).global_pose.to_np()[:3, 3],
        handle_position,
        atol=1e-12,
    )
    np.testing.assert_array_equal(physics.data.qpos, before)
    assert not np.allclose(physics.rack.pose[:3], captured[RackGeometry.BODY].position)
