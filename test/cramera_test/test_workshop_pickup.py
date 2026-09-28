"""
A bin pickup leaves space for nearby loose fasteners.
"""

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
    MovementType,
)
from coraplex.plans.factories import execute_single
from coraplex.plans.plan_node import MotionNode
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveToolCenterPointMotion,
)
from cramera.workshop_demo import ScrewGrasp
from cramera.workshop_pickup import PickUpLooseScrewAction
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.pr2 import PR2


# %% narrow approach and retained orientation
def test_loose_screw_pickup_presets_fingers_before_the_final_descent(pr2_world_copy):
    """
    The bin approach uses a bounded opening and lifts without releasing orientation.
    """
    world = pr2_world_copy
    [robot] = world.get_semantic_annotations_by_type(PR2)
    grasp = ScrewGrasp(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        robot.left_arm.end_effector,
        manipulation_offset=0.12,
    )
    action = PickUpLooseScrewAction(robot.root, Arms.LEFT, grasp)
    execute_single(action, context=Context(world=world, robot=robot))
    motions = [
        node.designator
        for node in action._action_plan.plan.nodes
        if isinstance(node, MotionNode)
    ]
    assert isinstance(motions[0], MoveToolCenterPointMotion)
    opening = motions[1]
    assert isinstance(opening, MoveGripperMotion)
    assert opening.motion is GripperState.OPEN
    assert opening.goal_state.target_values == [action.approach_finger_position]
    [joint] = opening.goal_state.connections
    opened = robot.left_arm.end_effector.get_joint_state_by_type(GripperState.OPEN)
    opened.apply_to(world)
    assert 0 < action.approach_finger_position < joint.position
    assert motions[-1].movement_type is MovementType.CARTESIAN
