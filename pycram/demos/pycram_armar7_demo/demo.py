import os
from pathlib import Path

from krrood.entity_query_language.factories import entity, an, variable, count
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.motion_executor import simulated_robot, simulated_robot_without_collision
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.utils import get_path_to_project_root
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
)
from semantic_digital_twin.world_description.utils import world_with_urdf_factory
import pycram

# %% Environment Setup
environment_path = os.path.join("package://iai_apartment/urdf/apartment.urdf")
# environment_path = os.path.join("package://iai_kit_mobile_lab/urdf/mobile_kitchen.urdf")
# environment_path = os.path.join("package://iai_kit_mobile_lab/urdf/R007.urdf")
environment_world = URDFParser.from_file(environment_path).parse()

# %% Robot Setup
robot_path = os.path.join("package://garmi_description/urdf/garmi.urdf")
robot_starting_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
    3.8,
    8.40,
    0,
)
robot_world = world_with_urdf_factory(robot_path, Garmi, OmniDrive, robot_starting_pose)

robot_world.merge_world(environment_world)
world = robot_world

# %% Spawn Objects
project_root = get_path_to_project_root(Path(pycram.__file__).resolve())
milk_world = STLParser(
    os.path.join(project_root, "resources", "objects", "milk.stl")
).parse()
cereal_world = STLParser(
    os.path.join(
        project_root,
        "resources",
        "objects",
        "breakfast_cereal.stl",
    )
).parse()
with world.modify_world():
    world.merge_world_at_pose(
        milk_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            5.3, 8.40, 1.09, reference_frame=world.root
        ),
    )
    world.merge_world_at_pose(
        cereal_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            5.3, 8.25, 1.09, reference_frame=world.root
        ),
    )

# %% Visualization
try:
    import rclpy

    rclpy.init()
    rclpy_node = rclpy.create_node("ros_node")
    viz = VizMarkerPublisher(_world=world, node=rclpy_node)
    viz.with_tf_publisher()
except ImportError:
    pass

# %% Demo
context = Context.from_world(world)
garmi = world.get_semantic_annotations_by_type(Garmi)[0]
milk_place_pose = Pose(Point3(x=2.2, y=7.6, z=0.865), reference_frame=world.root)

robot = variable(AbstractRobot, [garmi])
number_of_arms = an(entity(count(robot.manipulators))).tolist()

print(number_of_arms)
with simulated_robot_without_collision:
    sequential(
        [ParkArmsAction(arm=Arms.BOTH)],
        context,
    ).perform()
