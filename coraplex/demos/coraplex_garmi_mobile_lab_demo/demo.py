import os

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import OmniDrive

# %% World setup

lab_world = URDFParser.from_file("package://iai_kit_mobile_lab/urdf/R007.urdf").parse()
garmi_world = URDFParser.from_file(
    "package://garmi_description/urdf/garmi.urdf"
).parse()
bowl_world = STLParser(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "objects", "bowl.stl"
    )
).parse()

with lab_world.modify_world():
    root_connection = OmniDrive.create_with_dofs(
        parent=lab_world.root, child=garmi_world.root, world=lab_world
    )
    lab_world.merge_world(garmi_world, root_connection)
    root_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(3.6, 8.7, 0)
    lab_world.merge_world_at_pose(
        bowl_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            5.5, 8.6, 1.0, reference_frame=lab_world.root
        ),
    )

world = lab_world

# %% Visualization

try:
    import rclpy
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )

    rclpy.init()
    node = rclpy.create_node("viz_marker")
    VizMarkerPublisher(_world=world, node=node).with_tf_publisher()
except ImportError:
    node = None

# %% Demo

garmi = Garmi.from_world(world)
context = Context(world=world, robot=garmi, _debug=False, ros_node=node)
context.evaluate_conditions = False

plan = sequential(
    [
        ParkArmsAction(Arms.BOTH),
        MoveTorsoAction(TorsoState.HIGH),
        TransportAction(
            world.get_body_by_name("bowl.stl"),
            Pose.from_xyz_rpy(5.25, 11.9, 0.88, yaw=1.57, reference_frame=world.root),
            Arms.LEFT,
        ),
    ],
    context=context,
).plan

with simulated_robot:
    plan.perform()
