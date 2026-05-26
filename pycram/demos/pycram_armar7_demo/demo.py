import os
from pathlib import Path

from krrood.entity_query_language.backends import EntityQueryLanguageBackend
from krrood.entity_query_language.factories import (
    entity,
    an,
    variable,
    count,
    underspecified,
    variable_from,
)
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.language import SequentialNode
from pycram.motion_executor import (
    simulated_robot,
    simulated_robot_without_collision,
    simulated_robot_with_collision,
)
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.pick_up import PickUpAction
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
import matplotlib

matplotlib.use("TkAgg")
# Source - https://stackoverflow.com/a/46965602
# Posted by Amit Lohan, modified by community. See post 'Timeline' for change history
# Retrieved 2026-05-26, License - CC BY-SA 4.0

import matplotlib.pyplot as plt

# Source - https://stackoverflow.com/a/28154000
# Posted by DanT
# Retrieved 2026-05-26, License - CC BY-SA 3.0


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


def print_plans(plans):
    for i, root in enumerate(plans, start=1):
        plan = root.plan
        print(f"\nPlan {i}: {root.__class__.__name__}")
        print(f"  status: {root.status.name}")
        print(f"  nodes: {len(plan.nodes)}")

        for node in plan.nodes:
            indent = "  " * (node.depth + 1)
            print(f"{indent}- {node!r} [{node.status.name}]")


with world.modify_world():
    world.merge_world_at_pose(
        milk_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            2.37, 2, 1.05, reference_frame=world.root
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
milk_place_pose = Pose(Point3(2.37, 2, 1.05), reference_frame=world.root)

robot = variable(AbstractRobot, [garmi])
number_of_arms = an(entity(count(robot.manipulators))).tolist()


grasp_description = underspecified(GraspDescription)(
    approach_direction=ApproachDirection.FRONT,
    vertical_alignment=VerticalAlignment.NoAlignment,
    manipulator=variable_from([garmi.left_arm.manipulator]),
)
plan_generator = underspecified(sequential, target_type=SequentialNode)(
    children=[
        underspecified(NavigateAction)(
            target_location=(
                target_locations := variable_from(
                    [
                        Pose.from_xyz_quaternion(1, 2, 0, reference_frame=world.root),
                        Pose.from_xyz_quaternion(2, 1, 0, reference_frame=world.root),
                    ]
                )
            ),
            keep_joint_states=True,
        ),
        underspecified(PickUpAction)(
            arm=...,
            grasp_description=grasp_description,
            object_designator=world.get_body_by_name("milk.stl"),
        ),
    ],
    context=context,
)

plan_generator.resolve()
plans = list(EntityQueryLanguageBackend().evaluate(plan_generator))
print_plans(plans)
with simulated_robot_without_collision:
    plans[0].perform()
