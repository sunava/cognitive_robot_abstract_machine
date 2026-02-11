import os

import rclpy

from demos.thesis.simulation_setup import add_box, BoxSpec
from demos.thesis_new.frame_provider import WorldTransformFrameProvider
from demos.thesis_new.phase_presets import build_default_sequence, build_bowl_sequence
from demos.thesis_new.phase_models import Pose, FixedFrameProvider
from demos.thesis_new.rviz import PhaseSequenceRviz
from demos.thesis_new.world_utils import try_get_body, make_identity_pose_stamped, body_local_aabb
from pycram.datastructures.dataclasses import Context
from pycram.language import SequentialPlan
from pycram.process_module import simulated_robot
from pycram.robot_plans import MoveTorsoActionDescription
from pycram.testing import setup_world
from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Scale


def _setup_world():
    """Create a demo world with a bowl and a box."""
    world = setup_world()

    bowl = STLParser(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "objects", "bowl.stl"
        )
    ).parse()
    for shape in bowl.root.visual.shapes:
        shape.scale = Scale(x=3.5, y=3.5, z=3.5)
    for shape in bowl.root.collision.shapes:
        shape.scale = Scale(x=3.5, y=3.5, z=3.5)

    with world.modify_world():
        world.merge_world_at_pose(
            bowl,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                2.4, 2.2, 1, reference_frame=world.root
            ),
        )

    with world.modify_world():
        box = add_box(
            world, BoxSpec(name="muh_box", scale_xyz=(0.3, 0.1, 0.1)), tf_frame="/map"
        )
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=box,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.9, y=1.0, z=0.95
                ),
            )
        )

    return world


def main():
    """Run the RViz demo for default and bowl-constrained sequences."""
    world = _setup_world()

    rclpy.init()
    node = rclpy.create_node("pycram_demo")

    tf_wrapper = TFWrapper(node=node)
    TFPublisher(node=node, world=world)
    VizMarkerPublisher(world, node)

    tf_wrapper.wait_for_transform(
        "apartment/apartment_root",
        "pr2/base_footprint",
        timeout=RclpyDuration(seconds=1.0),
        time=Time(),
    )

    PR2.from_world(world)
    context = Context.from_world(world)

    with world.modify_world():
        world_reasoner = WorldReasoner(world)
        world_reasoner.reason()
        world.add_semantic_annotations(
            [
                Bowl(root=world.get_body_by_name("bowl.stl")),
            ]
        )

    plan = SequentialPlan(
        context,
        MoveTorsoActionDescription(TorsoState.HIGH),
    )
    with simulated_robot:
        plan.perform()

    seq = build_default_sequence()

    prov_world = FixedFrameProvider(Pose())
    _, P_world, id_world = seq.sample(prov_world, dt=0.01)

    rv = PhaseSequenceRviz(
        P_world,
        id_world,
        frame_id="apartment/apartment_root",
        topic="phase_sequence_markers",
        node=node,
    )
    rv.publish_once()

    bowl_body = try_get_body(world, "bowl.stl")
    if bowl_body is None:
        print("[info] body 'bowl.stl' not found, skipping object-dependent example.")
        return
    mins, maxs = body_local_aabb(
        bowl_body, use_visual=False, apply_shape_scale=False
    )
    print("AABB mins/maxs:", mins, maxs)
    seq_bowl = build_bowl_sequence(bowl_body, debug=True)

    seq_bowl = build_bowl_sequence(bowl_body, debug=True)
    prov_bowl = WorldTransformFrameProvider(
        world=world,
        source_frame=bowl_body,
        root_frame=world.root,
        make_identity_spatial=make_identity_pose_stamped,
    )
    _, P_bowl, id_bowl = seq_bowl.sample(prov_bowl, dt=0.01)

    rv_bowl = PhaseSequenceRviz(
        P_bowl,
        id_bowl,
        frame_id="apartment/apartment_root",
        topic="phase_sequence_markers_bowl",
        node=node,
    )
    rv_bowl.publish_once()


if __name__ == "__main__":
    main()
