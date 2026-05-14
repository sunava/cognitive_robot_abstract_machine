import rclpy

from demos.bachelor_thesis.classes_and_methods.helper_classes_and_methods import create_annotations_for_bodies_sage10k
from demos.bachelor_thesis.events.event_handler import EventDispatcher
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.locations.locations import CostmapLocation
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import sequential, execute_single
from pycram.plans.failures import BodyUnfetchable
from pycram.robot_plans.actions.core.navigation import NavigateAction
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader

from time import sleep
import threading

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import OmniDrive
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

#-- WORLD SETUP --------------------------------------------------------------------------------------------------------

rclpy.init()
loader = Sage10kDatasetLoader()

# Pick a scene
#urls = Sage10kDatasetLoader.available_scenes()
scene = loader.create_scene("https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/scenes/20251213_020526_layout_84b703fb.zip")
#print(urls[0])

# This builds the entire World object — rooms, walls, floors, doors, objects
world = scene.create_world()

node = rclpy.create_node("viz_marker")
v = VizMarkerPublisher(_world=world, node=node).with_tf_publisher()

# In separatem Thread spinnen, damit dein restlicher Code weiterläuft
spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
spin_thread.start()

#-- ROBOT SETUP --------------------------------------------------------------------------------------------------------
hsrb_sem_world = URDFParser.from_xacro(
        "package://hsr_description/robots/hsrb4s.urdf.xacro"
    ).parse()

with world.modify_world():
    hsrb_root = hsrb_sem_world.get_body_by_name("base_footprint")
    apartment_root = world.root
    c_root_bf = OmniDrive.create_with_dofs(
        parent=apartment_root, child=hsrb_root, world=world
    )
    world.merge_world(hsrb_sem_world, c_root_bf)
    c_root_bf.origin = HomogeneousTransformationMatrix.from_xyz_rpy(1, 0.4, 0)

hsrb = HSRB.from_world(world)
context = Context(world=world, robot=hsrb)

#-- ANNOTATION SETUP ---------------------------------------------------------------------------------------------------
dispatcher = EventDispatcher()

print("#" * 5, "BODIES", "#" * 100)
for bod in world.bodies:
    print(bod.name)

print("#" * 5, "ANNOTS", "#" * 100)
for ann in  world.semantic_annotations:
    print(ann.name, isinstance(ann, HasSupportingSurface), ann.__class__)

create_annotations_for_bodies_sage10k(world)

print("#" * 5, "ANNOTS AFTER CREATION", "#" * 85)
for ann in  world.semantic_annotations:
    print(ann.name, isinstance(ann, HasSupportingSurface), ann.__class__)

#-----------------------------------------------------------------------------------------------------------------------
context.evaluate_conditions = False

table_pose = None
for bod in world.bodies:
    if bod.name.prefix is not None:
        if "pouf" in bod.name.prefix:
            table_pose = bod.global_pose
            break

print("pouf", "x:", table_pose.x, "y:", table_pose.y, "z:", table_pose.z)
print("table ",table_pose.x)

with simulated_robot:
    nav_loc = CostmapLocation(
            target=table_pose,
            reachable_arm=Arms.LEFT,
            reachable=False,
            context=context,
    )

    print("nav ",nav_loc.ground())
            # Tries to find a pick-up position for the robot that uses the given arm

    nav_pose = nav_loc.ground()
    plan = execute_single(NavigateAction(nav_pose, True), context).plan

    if not nav_loc:
        raise Exception("booohooo")

    plan.perform()

    print("Roboter:", hsrb.root.global_pose.x, hsrb.root.global_pose.y, "frame: ", hsrb_root.global_pose.reference_frame)
    print("Pouf:", table_pose.x, table_pose.y, "frame: ", table_pose.reference_frame)
    print("nav_pose: ", nav_pose.x, nav_pose.y, "frame: ", nav_pose.reference_frame)

