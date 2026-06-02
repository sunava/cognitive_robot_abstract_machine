import os
from contextlib import contextmanager
from enum import Enum

from docutils.nodes import reference

from demos.bachelor_thesis.actions.random_location_generator import random_location_list, \
    pose_to_homogeneous_transformation_matrix_from_xyz_quaternion
from demos.bachelor_thesis.actions.simulate_perception import simulate_perception
from demos.bachelor_thesis.events.event_handler import EventDispatcher
from pycram import plans
from pycram.datastructures.enums import Arms
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import sequential, execute_single
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Spoon, Bottle, Cup, ShelfLayer, \
    CounterTop, Table, DishwasherTab
from semantic_digital_twin.spatial_types import Point3, Quaternion
from semantic_digital_twin.spatial_types.spatial_types import Pose, HomogeneousTransformationMatrix
from semantic_digital_twin.robots.hsrb import HSRB
from pycram.datastructures.dataclasses import Context
from demos.bachelor_thesis.hsrb_setup_world import hsrb_setup_world

from demos.bachelor_thesis.classes_and_methods.helper_classes_and_methods import Environment, perf_step, perf_print, \
    timed_plan, timed_parse_stl, debug_task_list_for_demo, print_sorted_task_list, sort_tasks, \
    compare_robot_world_with_real

environment = Environment.SuturoApartmentLab

#------------------ standard setup -------------------------------------------------------------------------------------
with perf_step("hsrb_setup_world"):
    world, dispatcher = hsrb_setup_world(environment=environment)

with perf_step("store known furniture"):
    dispatcher.known_furniture = world.bodies


#-----------------------------------------------------------------------------------------------------------------------
dispatcher.correct_location_tableware = world.get_semantic_annotation_by_name("counterTop")
dispatcher.correct_location_food = world.get_semantic_annotation_by_name("table")
dispatcher.correct_location_drinks = world.get_semantic_annotation_by_name("desk")
dispatcher.correct_location_all_other_items = world.get_semantic_annotation_by_name("shelf_1")

dispatcher.dining_table = world.get_semantic_annotation_by_name("dining_table")


#-----------------------------------------------------------------------------------------------------------------------



bowl = timed_parse_stl("bowl", "bowl.stl")

spoon = timed_parse_stl("spoon", "spoon.stl")

pitcher = timed_parse_stl("pitcher", "Static_MilkPitcher.stl")

coke = timed_parse_stl("coke", "Static_CokeBottle.stl")

jeroen_cup = timed_parse_stl("jeroen cup", "jeroen_cup.stl")

dishwasher_tab = timed_parse_stl("dishwasher tab", "dishwasher_tab.stl")



with perf_step("generate random object locations"):
    locs = random_location_list(world, 10)


with perf_step("modify world: place objects and supporting surfaces"):
    with world.modify_world():
        with perf_step("merge object meshes into world"):
            world.merge_world_at_pose(
                bowl,
                pose_to_homogeneous_transformation_matrix_from_xyz_quaternion(Pose(Point3(x=2, y=-1.6, z=0.57)), world),
            )
            world.merge_world_at_pose(
                spoon,
                pose_to_homogeneous_transformation_matrix_from_xyz_quaternion(locs[1], world),
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
                pose_to_homogeneous_transformation_matrix_from_xyz_quaternion(Pose(Point3(x=2, y=-1, z=0.14)), world),
            )
            world.merge_world_at_pose(
                dishwasher_tab,
                pose_to_homogeneous_transformation_matrix_from_xyz_quaternion(Pose(Point3(x=2, y=-1, z=0.14)), world),
            )

        with perf_step("add object semantic annotations"):
            world.add_semantic_annotations(
                [
                    Bowl(root=world.get_body_by_name("bowl.stl"), name=PrefixedName("bowl.stl")),
                    Spoon(root=world.get_body_by_name("spoon.stl"), name=PrefixedName("spoon.stl")),
                    Bottle(root=world.get_body_by_name("Static_MilkPitcher.stl"), name=PrefixedName("Static_MilkPitcher.stl")),
                    Bottle(root=world.get_body_by_name("Static_CokeBottle.stl"), name=PrefixedName("Static_CokeBottle.stl")),
                    Cup(root=world.get_body_by_name("jeroen_cup.stl"), name=PrefixedName("jeroen_cup.stl")),
                    DishwasherTab(root=world.get_body_by_name("dishwasher_tab.stl"), name=PrefixedName("dishwasher_tab.stl")),
                ]
            )
            # TODO: not many object stls so stuff like banana: yellow spoon, carrot: orange spoon, ...

        # world.add_semantic_annotations(
        #     [
        #         ShelfLayer(root=world.get_body_by_name("shelf_1"), name=PrefixedName("shelf_1")),
        #         ShelfLayer(root=world.get_body_by_name("shelf_2"), name=PrefixedName("shelf_2"))
        #
        #     ]
        # )
        with perf_step("collect supporting surface annotations"):
            supporting_surfaces = []
            supporting_surfaces.append(world.get_semantic_annotation_by_name("shelf_1"))
            supporting_surfaces.append(world.get_semantic_annotation_by_name("shelf_2"))
            supporting_surfaces.append(world.get_semantic_annotation_by_name("counterTop"))
            supporting_surfaces.append(world.get_semantic_annotation_by_name("table"))
            supporting_surfaces.append(world.get_semantic_annotation_by_name("lowerTable"))
            supporting_surfaces.append(world.get_semantic_annotation_by_name("desk"))
            supporting_surfaces.append(world.get_semantic_annotation_by_name("cooking_table"))
            supporting_surfaces.append(world.get_semantic_annotation_by_name("dining_table"))
            supporting_surfaces.append(world.get_semantic_annotation_by_name("dishwasher_rack"))




        for surface in supporting_surfaces:
            if isinstance(surface, HasSupportingSurface):
                with perf_step(f"calculate supporting surface: {surface.name}"):
                    surface.calculate_supporting_surface()


with perf_step("setup ROS visualization marker"):
    try:
        import rclpy
        try:
            rclpy.init()
        except:
            pass
        from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
            VizMarkerPublisher,
        )

        node = rclpy.create_node("viz_marker")
        v = VizMarkerPublisher(_world=world, node=node).with_tf_publisher()
    except ImportError:
        node = None

with perf_step("create HSRB robot from world"):
    hsrb = HSRB.from_world(world)

with perf_step("create execution context"):
    context = Context(world=world, robot=hsrb)

with perf_step("initial world reasoning"):
    with world.modify_world():
        world_reasoner = WorldReasoner(world)
        world_reasoner.reason()


context.evaluate_conditions = False

# get coordinates from publish point in rviz2
plan_labels = [
    "park left arm",
    "navigate dishwasher",
    "navigate kitchen counter 1",
    "navigate kitchen counter 2",
    "navigate high kitchen counter 1",
    "navigate high kitchen counter 2",
    "navigate transition before counter",
    "navigate pc desk",
    "navigate popcorn table",
    "navigate transition before wall",
    "navigate sofa table",
    "navigate living room table",
    "navigate shelf",
]

plan_driving = [
        timed_plan("park left arm", ParkArmsAction(Arms.LEFT), context),

        # dishwasher
        timed_plan("navigate dishwasher", NavigateAction(
            target_location=Pose(Point3(1.2266756, -0.2182769775390625, 0.0), orientation=(Quaternion(z=-0.316469, w=0.948602)),
                                reference_frame=world.root), keep_joint_states=True), context),

        # kitchen counter
        timed_plan("navigate kitchen counter 1", NavigateAction(
            target_location=Pose(Point3(1.677089, -0.91819, 0), orientation=(Quaternion(z=-0.673775, w=0.7389362)),
                                reference_frame=world.root), keep_joint_states=True), context),
        timed_plan("navigate kitchen counter 2", NavigateAction(
            target_location=Pose(Point3(3.099597, -0.897218, 0), orientation=(Quaternion(z=-0.679143, w=0.734005727)),
                                 reference_frame=world.root), keep_joint_states=True), context),

        # high kitchen counter
        timed_plan("navigate high kitchen counter 1", NavigateAction(
            target_location=Pose(Point3(3.393497, -0.3331599, 0), orientation=(Quaternion(z=0.748984068, w=0.6625880)),
                                 reference_frame=world.root), keep_joint_states=True), context),
        timed_plan("navigate high kitchen counter 2", NavigateAction(
            target_location=Pose(Point3(4.839765, -0.061004, 0), orientation=(Quaternion(z=0.7564081, w=0.654099959)),
                                 reference_frame=world.root), keep_joint_states=True), context),

        # transition point to not drive through counter
        timed_plan("navigate transition before counter", NavigateAction(
            target_location=Pose(Point3(1.777967, -0.090250, 0), orientation=(Quaternion(z=0.900024, w=0.435840186)),
                                 reference_frame=world.root), keep_joint_states=True), context),

        # pc desk
        timed_plan("navigate pc desk", NavigateAction(
            target_location=Pose(Point3(1.0679969, 1.530962, 0), orientation=(Quaternion(z=-0.9981287, w=0.0611478)),
                                                reference_frame=world.root), keep_joint_states=True), context),

        # popcorn table
        timed_plan("navigate popcorn table", NavigateAction(
            target_location=Pose(Point3(1.09943246, 5.53489685, 0), orientation=(Quaternion(z=0.7474030, w=0.6643709)),
                                                reference_frame=world.root), keep_joint_states=True), context),

        # transition point to not drive in wall
        timed_plan("navigate transition before wall", NavigateAction(
            target_location=Pose(Point3(1.7850532, 3.3190565, 0), orientation=(Quaternion(z=0.1006678, w=0.99492009)),
                                                reference_frame=world.root), keep_joint_states=True), context),

        # sofa table
        timed_plan("navigate sofa table", NavigateAction(
            target_location=Pose(Point3(3.57369399, 3.0707988, 0), orientation=(Quaternion(z=0.0701156, w=0.99753887)),
                                 reference_frame=world.root), keep_joint_states=True), context),

        # living room table
        timed_plan("navigate living room table", NavigateAction(
            target_location=Pose(Point3(3.2095706, 6.522722, 0), orientation=(Quaternion(z=-0.9995140, w=0.03117147)),
                                 reference_frame=world.root), keep_joint_states=True), context),

        # shelf
        timed_plan("navigate shelf", NavigateAction(
            target_location=Pose(Point3(3.3593473, 5.40832, 0), orientation=(Quaternion(z=0.0721225, w=0.997395779)),
                                 reference_frame=world.root), keep_joint_states=True), context),
        ]



with perf_step("execute all plans with simulated robot"):
    with simulated_robot:
        for index, (label, plan) in enumerate(zip(plan_labels, plan_driving), start=1):
            step_label = f"{index:02d}/{len(plan_driving)} {label}"
            with perf_step(f"perform plan: {step_label}"):
                plan.perform()
            with perf_step(f"simulate perception after: {step_label}"):
                visible_bodies = simulate_perception(
                    world,
                    dispatcher,
                    context,
                    hsrb,
                    perf_label=f"perception {step_label}",
                )
                visible_count = len(visible_bodies) if visible_bodies is not None else 0
                perf_print(f"visible bodies after {step_label}: {visible_count}")

        debug_task_list_for_demo(dispatcher)


perf_print("done")
print_sorted_task_list(sort_tasks(dispatcher.activated_tasks, 300), 300)

res = compare_robot_world_with_real(dispatcher, world)
print(res)
