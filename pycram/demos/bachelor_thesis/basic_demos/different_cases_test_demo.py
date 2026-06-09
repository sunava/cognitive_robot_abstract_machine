import os

import rustworkx

from demos.bachelor_thesis.actions.predicate_mock import misplaced
from demos.bachelor_thesis.actions.random_location_generator import random_location_list, \
    pose_to_homogeneous_transformation_matrix_from_xyz_quaternion
from demos.bachelor_thesis.classes_and_methods.tasks import PutAwayObjectTask, SetTableTask, CleanTableTask
from krrood.entity_query_language.factories import underspecified, variable
from pycram import plans
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.locations.locations import CostmapLocation
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import sequential, execute_single
from pycram.robot_plans.actions.core.misc import MoveToReach
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.adapters.ros.visualization.viz_marker import CostmapHeatmapRviz
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import PointOccupiedError
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Spoon, Bottle, Cup, ShelfLayer, \
    CounterTop
from semantic_digital_twin.spatial_types import Point3, Quaternion
from semantic_digital_twin.spatial_types.spatial_types import Pose, HomogeneousTransformationMatrix, Pose2D
from semantic_digital_twin.robots.hsrb import HSRB
from pycram.datastructures.dataclasses import Context
from demos.bachelor_thesis.hsrb_setup_world import hsrb_setup_world
from demos.bachelor_thesis.classes_and_methods.helper_classes_and_methods import Environment
from semantic_digital_twin.world_description.graph_of_convex_sets import navigation_map_at_target, \
    translate_free_space_to_where_condition

#------------------ standard setup -------------------------------------------------------------------------------------
world, dispatcher = hsrb_setup_world(Environment.SuturoApartmentLab)



bowl = STLParser(
    os.path.join(
        os.path.dirname(__file__), "../..", "..", "resources", "objects", "bowl.stl"
    )
).parse()

spoon = STLParser(
    os.path.join(
        os.path.dirname(__file__), "../..", "..", "resources", "objects", "spoon.stl"
    )
).parse()

pitcher = STLParser(
os.path.join(
        os.path.dirname(__file__), "../..", "..", "resources", "objects", "Static_MilkPitcher.stl"
    )
).parse()

coke = STLParser(
os.path.join(
        os.path.dirname(__file__), "../..", "..", "resources", "objects", "Static_CokeBottle.stl"
    )
).parse()

jeroen_cup = STLParser(
os.path.join(
        os.path.dirname(__file__), "../..", "..", "resources", "objects", "jeroen_cup.stl"
    )
).parse()

locs = random_location_list(world, 10)


with world.modify_world():
    world.merge_world_at_pose(
        bowl,
        pose_to_homogeneous_transformation_matrix_from_xyz_quaternion(Pose(Point3(x=2, y=-1.6, z=0.57)), world),
    )
    world.merge_world_at_pose(
        spoon,
        pose_to_homogeneous_transformation_matrix_from_xyz_quaternion(Pose(Point3(x=3, y=1, z=0.82)), world),
    )
    world.merge_world_at_pose(
        pitcher,
        pose_to_homogeneous_transformation_matrix_from_xyz_quaternion(locs[2], world),
    )
    world.merge_world_at_pose(
        coke,
        pose_to_homogeneous_transformation_matrix_from_xyz_quaternion(locs[3], world),
    )
    world.merge_world_at_pose(
        jeroen_cup,
        pose_to_homogeneous_transformation_matrix_from_xyz_quaternion(locs[4], world),
    )
    world.add_semantic_annotations(
        [
            Bowl(root=world.get_body_by_name("bowl.stl"), name=PrefixedName("bowl.stl")),
            Spoon(root=world.get_body_by_name("spoon.stl"), name=PrefixedName("spoon.stl")),
            Bottle(root=world.get_body_by_name("Static_MilkPitcher.stl"), name=PrefixedName("Static_MilkPitcher.stl")),
            Bottle(root=world.get_body_by_name("Static_CokeBottle.stl"), name=PrefixedName("Static_CokeBottle.stl")),
            Cup(root=world.get_body_by_name("jeroen_cup.stl"), name=PrefixedName("jeroen_cup.stl")),
        ]
    )
    # world.add_semantic_annotations(
    #     [
    #         ShelfLayer(root=world.get_body_by_name("shelf_1"), name=PrefixedName("shelf_1")),
    #         ShelfLayer(root=world.get_body_by_name("shelf_2"), name=PrefixedName("shelf_2"))
    #
    #     ]
    # )
    supporting_surfaces = []
    supporting_surfaces.append(world.get_semantic_annotation_by_name("shelf_1"))
    supporting_surfaces.append(world.get_semantic_annotation_by_name("shelf_2"))
    supporting_surfaces.append(world.get_semantic_annotation_by_name("counterTop"))
    supporting_surfaces.append(world.get_semantic_annotation_by_name("table"))
    supporting_surfaces.append(world.get_semantic_annotation_by_name("lowerTable"))
    supporting_surfaces.append(world.get_semantic_annotation_by_name("desk"))
    supporting_surfaces.append(world.get_semantic_annotation_by_name("cooking_table"))
    supporting_surfaces.append(world.get_semantic_annotation_by_name("dining_table"))

    for surface in supporting_surfaces:
        if isinstance(surface, HasSupportingSurface):
            surface.calculate_supporting_surface()


try:
    import rclpy
    try:
        rclpy.init()
    except:
        pass
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher, CostmapHeatmapRviz,
)

    node = rclpy.create_node("viz_marker")
    v = VizMarkerPublisher(_world=world, node=node).with_tf_publisher()
except ImportError:
    node = None

hsrb = HSRB.from_world(world)
context = Context(world=world, robot=hsrb)

with world.modify_world():
    world_reasoner = WorldReasoner(world)
    world_reasoner.reason()


context.evaluate_conditions = False

# get coordinates from publish point in rviz2
plan_driving = [execute_single(ParkArmsAction(Arms.LEFT), context).plan,

        # kitchen counter
        execute_single(NavigateAction(
            target_location=Pose(Point3(1.677089, -0.91819, 0), orientation=(Quaternion(z=-0.673775, w=0.7389362)),
                                reference_frame=world.root), keep_joint_states=True), context).plan,
        execute_single(NavigateAction(
            target_location=Pose(Point3(3.099597, -0.897218, 0), orientation=(Quaternion(z=-0.679143, w=0.734005727)),
                                 reference_frame=world.root), keep_joint_states=True),context).plan,

        # high kitchen counter
        execute_single(NavigateAction(
            target_location=Pose(Point3(3.393497, -0.3331599, 0), orientation=(Quaternion(z=0.748984068, w=0.6625880)),
                                 reference_frame=world.root), keep_joint_states=True), context).plan,
        execute_single(NavigateAction(
            target_location=Pose(Point3(4.839765, -0.061004, 0), orientation=(Quaternion(z=0.7564081, w=0.654099959)),
                                 reference_frame=world.root), keep_joint_states=True), context).plan,

        # transition point to not drive through counter
        execute_single(NavigateAction(
            target_location=Pose(Point3(1.777967, -0.090250, 0), orientation=(Quaternion(z=0.900024, w=0.435840186)),
                                 reference_frame=world.root), keep_joint_states=True), context).plan,

        # pc desk
        execute_single(NavigateAction(
            target_location=Pose(Point3(1.0679969, 1.530962, 0), orientation=(Quaternion(z=-0.9981287, w=0.0611478)),
                                                reference_frame=world.root), keep_joint_states=True),context).plan,

        # popcorn table
        execute_single(NavigateAction(
            target_location=Pose(Point3(1.09943246, 5.53489685, 0), orientation=(Quaternion(z=0.7474030, w=0.6643709)),
                                                reference_frame=world.root), keep_joint_states=True), context).plan,

        # transition point to not drive in wall
        execute_single(NavigateAction(
            target_location=Pose(Point3(1.7850532, 3.3190565, 0), orientation=(Quaternion(z=0.1006678, w=0.99492009)),
                                                reference_frame=world.root), keep_joint_states=True), context).plan,

        # sofa table
        execute_single(NavigateAction(
            target_location=Pose(Point3(3.57369399, 3.0707988, 0), orientation=(Quaternion(z=0.0701156, w=0.99753887)),
                                 reference_frame=world.root), keep_joint_states=True), context).plan,

        # living room table
        execute_single(NavigateAction(
            target_location=Pose(Point3(3.2095706, 6.522722, 0), orientation=(Quaternion(z=-0.9995140, w=0.03117147)),
                                 reference_frame=world.root), keep_joint_states=True), context).plan,

        # shelf
        execute_single(NavigateAction(
            target_location=Pose(Point3(3.3593473, 5.40832, 0), orientation=(Quaternion(z=0.0721225, w=0.997395779)),
                                 reference_frame=world.root), keep_joint_states=True), context).plan,
        ]

# def navigate_to_reach(object_body):
#     gcs = navigation_map_at_target(target=object_body)
#
#     arm = Arms.LEFT
#
#     min_p = object_body.collision.min_point
#     max_p = object_body.collision.max_point
#
#     x = min_p.x - 0.05
#     y = (min_p.y + max_p.y) / 2
#     z = (min_p.z + max_p.z) / 2
#
#     pre_grasp_pose = Pose.from_xyz_rpy(
#         x=x, y=y, z=z, reference_frame=object_body
#     )
#
#     # Find a node in free space that is near the pre-grasp pose.
#     target_node = gcs.node_of_point(pre_grasp_pose.position)
#     if target_node is None:
#         raise PointOccupiedError(
#            world.transform(pre_grasp_pose, world.root).position
#         )
#
#     gcs = gcs.create_subgraph(
#         list(
#             rustworkx.node_connected_component(
#                 gcs.graph, gcs.box_to_index_map[target_node]
#             )
#         )
#     )
#
#     reach_query = underspecified(MoveToReach)(
#         target_pose_offset_robot=underspecified(Pose2D)(
#             x=..., y=..., yaw=..., reference_frame=None
#         ),
#         hip_rotation=0.0,
#         target_pose_end_effector=pre_grasp_pose,
#         grasp_description=underspecified(GraspDescription)(
#             approach_direction=ApproachDirection.FRONT,
#             vertical_alignment=VerticalAlignment.NoAlignment,
#             end_effector=variable(EndEffector, world.semantic_annotations),
#             rotate_gripper=False,
#         ),
#     )
#
#     where_condition = translate_free_space_to_where_condition(
#         gcs.free_space_event,
#         reach_query.expression,
#         x_variable_name="MoveToReach.target_pose_offset_robot.x",
#         y_variable_name="MoveToReach.target_pose_offset_robot.y",
#     )
#
#     reach_action = reach_query.where(where_condition)
#
#
#     plan = execute_single(reach_action, context=context)
#     with simulated_robot:
#         plan.perform()
#


with simulated_robot:
    test_task = "set_table_task_dining_table"

    dispatcher.correct_location_tableware_dirty = world.get_semantic_annotation_by_name("counterTop")
    dispatcher.correct_location_tableware_clean = world.get_semantic_annotation_by_name("desk")
    dispatcher.correct_location_food = world.get_semantic_annotation_by_name("table")
    dispatcher.correct_location_drinks = world.get_semantic_annotation_by_name("desk")
    dispatcher.correct_location_all_other_items = world.get_semantic_annotation_by_name("shelf_1")

    dispatcher.dining_table = world.get_semantic_annotation_by_name("dining_table")

    surface_cache = {}
    cL = CostmapLocation(
        target=world.get_body_by_name("jeroen_cup.stl").global_pose,
        reachable=True,
        reachable_arm=None,
        # validate_reachability=False,
        # samples=1000,
        context=context,
        allowed_area_points=[
            Point3(x=0.3, y=-2.0, z=0.0),
            Point3(x=5.45, y=-1.69, z=0.0),
            Point3(x=4.0, y=6.45, z=0.0),
            Point3(x=-0.6, y=6.03, z=0.0),
        ],
    )
    costmap = cL.setup_costmaps(cL.target, cL.visible, cL.reachable)
    costmap_viz = CostmapHeatmapRviz(
        costmap,
        node=node,
        topic="costmap_heatmap",
        frame_id=str(world.root.name),
        sample_stride=2,
        min_normalized_value=0.0,
        republish_hz=None,  # publish once, no ROS spin needed
    )
    pickup_pose = cL.resolve()
    sequential(
        [NavigateAction(pickup_pose, True, teleport=True)],
        context,
    ).perform()
    # navigate_to_reach(world.get_body_by_name("bowl.stl"))
    # for plan in plan_driving:
    #     plan.perform()
    #     simulate_perception(world, dispatcher, context, hsrb)
    dispatcher.trigger_event([world.get_body_by_name("bowl.stl"), world.get_body_by_name("spoon.stl")], world)
    task1 = None

    for task in dispatcher.activated_tasks:
        if task.name == test_task:
            task1=task
