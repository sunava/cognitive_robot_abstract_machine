"""
Loaded carriers retain their contents through native transport plans.
"""

from __future__ import annotations

import numpy as np

from coraplex.datastructures.enums import ApproachDirection, VerticalAlignment
from cramera.workshop_transport import BoxHandleGrasp, WorkshopTransportDemo
from cramera.workshop_world import WorkshopWorld
from semantic_digital_twin.spatial_types import Pose
from semantic_digital_twin.world_description.connections import FixedConnection

from .test_workshop_world import workshop_bundle
from .test_workshop_bundle import workshop_directory


# %% grasp and mobile context
def test_box_grasp_approaches_handle_from_front(workshop_bundle):
    """
    The shelf approach clears the front wall and lifts before withdrawal.
    """
    demo = WorkshopTransportDemo(
        workshop=WorkshopWorld(bundle_directory=workshop_bundle)
    )
    world = demo.workshop.build()
    context = demo.build_context(world)
    box = world.get_body_by_name("screw_box")
    grasp = BoxHandleGrasp(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        context.robot.left_arm.end_effector,
    )
    before, contact, after = grasp.pose_sequence(Pose(reference_frame=box), box)
    np.testing.assert_allclose(
        contact.to_np()[:3, 3], box.global_pose.to_np()[:3, 3] + grasp.handle_offset
    )
    assert before.y < contact.y
    assert after.z > contact.z
    np.testing.assert_allclose(contact.to_np()[:3, 0], [0, 1, 0], atol=1e-12)


def test_transport_keeps_mobile_base_but_arm_tasks_do_not_drive_it(workshop_bundle):
    """
    Only explicit navigation actions may change the workshop base pose.
    """
    demo = WorkshopTransportDemo(
        workshop=WorkshopWorld(bundle_directory=workshop_bundle)
    )
    context = demo.build_context(demo.workshop.build())
    assert not isinstance(context.robot.root.parent_connection, FixedConnection)
    assert context.robot.mobile_base.full_body_controlled is False
