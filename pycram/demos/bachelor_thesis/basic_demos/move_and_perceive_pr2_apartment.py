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
from semantic_digital_twin.world_description.geometry import Color, Scale
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Spoon, Bottle, Cup, ShelfLayer, \
    CounterTop, Table, Wardrobe, Cabinet, Oven
from semantic_digital_twin.spatial_types import Point3, Quaternion
from semantic_digital_twin.spatial_types.spatial_types import Pose, HomogeneousTransformationMatrix
from semantic_digital_twin.robots.hsrb import HSRB
from pycram.datastructures.dataclasses import Context
from demos.bachelor_thesis.hsrb_setup_world import hsrb_setup_world
from time import sleep

from demos.bachelor_thesis.classes_and_methods.helper_classes_and_methods import Environment, perf_step, perf_print, \
    timed_plan, timed_parse_stl, debug_task_list_for_demo



environment = Environment.Pr2ApartmentLab

#------------------ standard setup -------------------------------------------------------------------------------------
with perf_step("hsrb_setup_world"):
    world, dispatcher = hsrb_setup_world(environment=environment)

with world.modify_world():
    dishwasher_rack = Table.create_with_new_body_in_world(
                    world=world,
                    name=PrefixedName("dishwasher_rack"),
                    world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.96, y=3.95, z=0.07),
                    scale=Scale(x=0.744, y=0.650, z=0.14)
                )
    for color in dishwasher_rack.bodies[0].visual.shapes:
        color.color = Color.RED()

with perf_step("store known furniture"):
    dispatcher.known_furniture = world.bodies

with world.modify_world():
    world.add_semantic_annotations(
        [
            CounterTop(root=world.get_body_by_name("countertop"), name=PrefixedName("counter")),
            Table(root=world.get_body_by_name("coffee_table"), name=PrefixedName("coffee_table")),
            Table(root=world.get_body_by_name("bedside_table"), name=PrefixedName("bedside_table")),
            CounterTop(root=world.get_body_by_name("cooktop"), name=PrefixedName("cooktop")),

        ]
    )


#-----------------------------------------------------------------------------------------------------------------------
dispatcher.dining_table = world.get_semantic_annotation_by_name("coffee_table")
dispatcher.correct_location_food = world.get_semantic_annotation_by_name("counter")
dispatcher.correct_location_drinks = world.get_semantic_annotation_by_name("cooktop")
dispatcher.correct_location_tableware = world.get_semantic_annotation_by_name("bedside_table")
dispatcher.correct_location_all_other_items = world.get_semantic_annotation_by_name("counter")


#-----------------------------------------------------------------------------------------------------------------------



bowl = timed_parse_stl("bowl", "bowl.stl")

spoon = timed_parse_stl("spoon", "spoon.stl")

pitcher = timed_parse_stl("pitcher", "Static_MilkPitcher.stl")

coke = timed_parse_stl("coke", "Static_CokeBottle.stl")

jeroen_cup = timed_parse_stl("jeroen cup", "jeroen_cup.stl")



with perf_step("generate random object locations"):
    locs = random_location_list(world, 10)

# print("generated_locations: ")
# i=0
# for loc in locs:
#     print(f"loc[{i}]: x={loc.x}, y={loc.y}, z={loc.z}")
#     i += 1

with perf_step("modify world: place objects and supporting surfaces"):
    with world.modify_world():
        with perf_step("merge object meshes into world"):
            world.merge_world_at_pose(
                bowl,
                pose_to_homogeneous_transformation_matrix_from_xyz_quaternion(locs[0], world),
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
                pose_to_homogeneous_transformation_matrix_from_xyz_quaternion(locs[4], world),
            )

        with perf_step("add object semantic annotations"):
            world.add_semantic_annotations(
                [
                    Bowl(root=world.get_body_by_name("bowl.stl"), name=PrefixedName("bowl.stl")),
                    Spoon(root=world.get_body_by_name("spoon.stl"), name=PrefixedName("spoon.stl")),
                    Bottle(root=world.get_body_by_name("Static_MilkPitcher.stl"), name=PrefixedName("Static_MilkPitcher.stl")),
                    Bottle(root=world.get_body_by_name("Static_CokeBottle.stl"), name=PrefixedName("Static_CokeBottle.stl")),
                    Cup(root=world.get_body_by_name("jeroen_cup.stl"), name=PrefixedName("jeroen_cup.stl")),
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
            supporting_surfaces.append(world.get_semantic_annotation_by_name("counter"))
            supporting_surfaces.append(world.get_semantic_annotation_by_name("coffee_table"))
            supporting_surfaces.append(world.get_semantic_annotation_by_name("bedside_table"))
            supporting_surfaces.append(world.get_semantic_annotation_by_name("cooktop"))




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
    "navigate kitchen counter",
    "navigate coffee machine",
    "navigate counter and dishwasher",
    "navigate table",
    "navigate tables by sofas",
]

plan_driving = [
        timed_plan("park left arm", ParkArmsAction(Arms.LEFT), context),

        timed_plan("navigate kitchen counter", NavigateAction(
            target_location=Pose(Point3(1.38141989, 0.80522632, 0.0), orientation=(Quaternion(z=0.29904239, w=0.954239825)),
                                reference_frame=world.root), keep_joint_states=True), context),

        timed_plan("navigate coffee machine", NavigateAction(
            target_location=Pose(Point3(2.14789, 2.86539, 0), orientation=(Quaternion(z=0.997806, w=0.066198992)),
                                 reference_frame=world.root), keep_joint_states = True), context),

        timed_plan("navigate counter and dishwasher", NavigateAction(
            target_location=Pose(Point3(0.9455279, 3.348699, 0), orientation=(Quaternion(z=0.1106727, w=0.9938569)),
                                reference_frame=world.root), keep_joint_states=True), context),

        timed_plan("navigate table", NavigateAction(
            target_location=Pose(Point3(2.8606681823, 4.7776088, 0), orientation=(Quaternion(z=-0.125561, w=0.992085804)),
                                 reference_frame=world.root), keep_joint_states=True), context),

        timed_plan("navigate tables by sofas", NavigateAction(
            target_location=Pose(Point3(15.391745, 2.0624361, 0), orientation=(Quaternion(z=0.0870467, w=0.99620422)),
                                 reference_frame=world.root), keep_joint_states=True), context),

        ]



with perf_step("execute all plans with simulated robot"):
    with simulated_robot:
        for index, (label, plan) in enumerate(zip(plan_labels, plan_driving), start=1):
            step_label = f"{index:02d}/{len(plan_driving)} {label}"
            print(step_label)
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
