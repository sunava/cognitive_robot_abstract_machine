from dataclasses import dataclass

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from coraplex.testing import setup_world
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class ShuttleRoute:
    """
    Round trips of one object between two poses.

    Each round trip transports the object from its current location to
    :attr:`pose_b` and then back to :attr:`pose_a`.
    """

    transported_object: Body
    """
    The object that shuttles between the two poses.
    """

    pose_a: Pose
    """
    The pose the object starts at and returns to.
    """

    pose_b: Pose
    """
    The pose the object is brought to before returning.
    """

    arm: Arms
    """
    The arm used for every transport.
    """

    round_trips: int = 2
    """
    How many times the object travels to :attr:`pose_b` and back.
    """

    def transport_actions(self) -> list[TransportAction]:
        """
        :return: The alternating transports that realize the round trips.
        """
        actions = []
        for _ in range(self.round_trips):
            actions.append(
                TransportAction(self.transported_object, self.pose_b, self.arm)
            )
            actions.append(
                TransportAction(self.transported_object, self.pose_a, self.arm)
            )
        return actions


world = setup_world()

try:
    import rclpy

    rclpy.init()
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )

    node = rclpy.create_node("viz_marker")
    visualization = VizMarkerPublisher(_world=world, node=node).with_tf_publisher()
except ImportError:
    node = None

pr2 = PR2.from_world(world)
context = Context(world=world, robot=pr2, _debug=False, ros_node=node)

with world.modify_world():
    world_reasoner = WorldReasoner(world)
    world_reasoner.reason()

context.evaluate_conditions = False

shuttle_route = ShuttleRoute(
    transported_object=world.get_body_by_name("milk.stl"),
    pose_a=Pose.from_xyz_rpy(2.37, 2, 1.05, reference_frame=world.root),
    pose_b=Pose.from_xyz_rpy(4.9, 3.3, 0.8, yaw=1.57, reference_frame=world.root),
    arm=Arms.LEFT,
)

plan = sequential(
    [
        ParkArmsAction(Arms.BOTH),
        MoveTorsoAction(TorsoState.HIGH),
        *shuttle_route.transport_actions(),
    ],
    context=context,
).plan

with simulated_robot:
    plan.perform()
