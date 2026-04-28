import os
from pycram.datastructures.enums import Arms
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.spatial_types import Point3, Quaternion
from semantic_digital_twin.spatial_types.spatial_types import Pose, HomogeneousTransformationMatrix
from semantic_digital_twin.robots.hsrb import HSRB
from pycram.datastructures.dataclasses import Context
from demos.bachelor_thesis.hsrb_setup_world import hsrb_setup_world



#------------------ standard setup -------------------------------------------------------------------------------------
world = hsrb_setup_world()[0]

spoon = STLParser(
    os.path.join(
        os.path.dirname(__file__), "../..", "..", "resources", "objects", "spoon.stl"
    )
).parse()
bowl = STLParser(
    os.path.join(
        os.path.dirname(__file__), "../..", "..", "resources", "objects", "bowl.stl"
    )
).parse()

with world.modify_world():
    world.merge_world_at_pose(
        bowl,
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            2.4, 2.2, 1, reference_frame=world.root
        ),
    )



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

hsrb = HSRB.from_world(world)
context = Context(world=world, robot=hsrb)

with world.modify_world():
    world_reasoner = WorldReasoner(world)
    world_reasoner.reason()


context.evaluate_conditions = False

# get coordinates from publish point in rviz2
plan_drive_around = sequential(
    [
        ParkArmsAction(Arms.LEFT),
        # kitchen counter
        NavigateAction(
            target_location=Pose(Point3(1.677089, -0.91819, 0), orientation=(Quaternion(z=-0.673775, w=0.7389362)),
                                            reference_frame=world.root), keep_joint_states=True),
        NavigateAction(
            target_location=Pose(Point3(3.099597, -0.897218, 0), orientation=(Quaternion(z=-0.679143, w=0.734005727)),
                                 reference_frame=world.root), keep_joint_states=True),

        # high kitchen counter
        NavigateAction(
            target_location=Pose(Point3(3.393497, -0.3331599, 0), orientation=(Quaternion(z=0.748984068, w=0.6625880)),
                                 reference_frame=world.root), keep_joint_states=True),
        NavigateAction(
            target_location=Pose(Point3(4.839765, -0.061004, 0), orientation=(Quaternion(z=0.7564081, w=0.654099959)),
                                 reference_frame=world.root), keep_joint_states=True),

        # transition point to not drive through counter
        NavigateAction(
            target_location=Pose(Point3(1.777967, -0.090250, 0), orientation=(Quaternion(z=0.900024, w=0.435840186)),
                                 reference_frame=world.root), keep_joint_states=True),

        # pc desk
        NavigateAction(
            target_location=Pose(Point3(1.0679969, 1.530962, 0), orientation=(Quaternion(z=-0.9981287, w=0.0611478)),
                                                reference_frame=world.root), keep_joint_states=True),
        # popcorn table
        NavigateAction(
            target_location=Pose(Point3(1.09943246, 5.53489685, 0), orientation=(Quaternion(z=0.7474030, w=0.6643709)),
                                                reference_frame=world.root), keep_joint_states=True),

        # transition point to not drive in wall
        NavigateAction(
            target_location=Pose(Point3(1.7850532, 3.3190565, 0), orientation=(Quaternion(z=0.1006678, w=0.99492009)),
                                                reference_frame=world.root), keep_joint_states=True),

        # sofa table
        NavigateAction(
            target_location=Pose(Point3(3.57369399, 3.0707988, 0), orientation=(Quaternion(z=0.0701156, w=0.99753887)),
                                 reference_frame=world.root), keep_joint_states=True),

        # living room table
        NavigateAction(
            target_location=Pose(Point3(3.2095706, 6.522722, 0), orientation=(Quaternion(z=-0.9995140, w=0.03117147)),
                                 reference_frame=world.root), keep_joint_states=True),

        # shelf
        NavigateAction(
            target_location=Pose(Point3(3.3593473, 5.40832, 0), orientation=(Quaternion(z=0.0721225, w=0.997395779)),
                                 reference_frame=world.root), keep_joint_states=True),

    ],
    context=context,
).plan

with simulated_robot:
    plan_drive_around.perform()