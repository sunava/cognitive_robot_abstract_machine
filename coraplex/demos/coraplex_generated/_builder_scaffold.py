"""Idle scene for the Teleoperation page (cramera)."""
import os
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, VisualizationBackend
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.visualization import WorldVisualization
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

_WORLDS = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "worlds")


def build_world(env_file, robot_xy):
    robot_world = URDFParser.from_file(PR2.get_ros_file_path()).parse()
    world = URDFParser.from_file(os.path.join(_WORLDS, env_file)).parse()
    with world.modify_world():
        robot_root = robot_world.get_body_by_name(PR2._get_root_body_name())
        drive = PR2.get_drive_connection_type().create_with_dofs(
            parent=world.root, child=robot_root, world=world)
        world.merge_world(robot_world, drive)
        drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(robot_xy[0], robot_xy[1], 0)
    standing = max(0.0, -world.height_of_lowest_collision_point_of_branch(robot_root))
    with world.modify_world():
        drive.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(
            z=standing, reference_frame=world.root)
    return world


world = build_world("kitchen.urdf", (0.0, 0.0))
visualization = WorldVisualization.from_environment(
    world, default_backend=VisualizationBackend.CRAMERA).start()
pr2 = PR2.from_world(world)
context = Context(world=world, robot=pr2, _debug=False, ros_node=visualization.ros_node)
with world.modify_world():
    WorldReasoner(world).reason()
context.evaluate_conditions = False
plan = sequential([ParkArmsAction(Arms.BOTH)], context=context).plan
visualization.attach_plan(plan)
with simulated_robot:
    plan.perform()
