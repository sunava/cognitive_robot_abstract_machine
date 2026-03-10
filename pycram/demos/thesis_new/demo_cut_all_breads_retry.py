import csv
import os
import numpy as np
import rclpy

from krrood.ormatic.dao import to_dao
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.datastructures.pose import PoseStamped
from pycram.designators.location_designator import CostmapLocation
from pycram.language import SequentialPlan
from pycram.motion_executor import (
    simulated_robot,
    simulated_robot_without_collision,
    simulated_robot_with_collision,
)
from pycram.robot_plans import (
    MoveTorsoActionDescription,
    ParkArmsActionDescription,
    CuttingActionDescription,
    NavigateActionDescription,
)
from pycram.tf_transformations import quaternion_from_euler, quaternion_multiply
from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    AvoidSelfCollisions,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.semantic_annotations.semantic_annotations import Knife
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color

from demos.thesis_new.demo_random_breads import setup_random_bread_world
import os
import numpy as np
from sqlalchemy import text

import rclpy
from trimesh.proximity import nearby_faces, closest_point

import pycram
from demos.thesis.simulation_setup import add_box, BoxSpec
from demos.thesis_new.thesis_math.world_utils import try_get_body
from krrood.ormatic.dao import to_dao
from krrood.ormatic.utils import drop_database
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot
from pycram.orm.ormatic_interface import Base
from pycram.orm.utils import pycram_sessionmaker
from pycram.robot_plans import (
    MoveTorsoActionDescription,
    MixingActionDescription,
    ParkArmsActionDescription,
    NavigateActionDescription,
    CuttingActionDescription,
    WipingActionDescription,
)

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
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Knife, Whisk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color, Scale
from semantic_digital_twin.world_description.world_entity import Body

RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "resources")
)
DEFAULT_BREAD_COLOR = Color(R=0.76, G=0.60, B=0.42)
ACTIVE_BREAD_COLOR = Color(R=0.52, G=0.82, B=0.98)
FAILED_BREAD_COLOR = Color(R=0.95, G=0.20, B=0.20)
SUCCESS_BREAD_COLOR = Color(R=0.62, G=0.92, B=0.62)
RESULTS_CSV_PATH = os.path.abspath("cut_all_breads_results.csv")


def _parse_stl(*relative_path_parts):
    return STLParser(os.path.join(RESOURCES_DIR, *relative_path_parts)).parse()


def _body_name(body):
    maybe_name = getattr(body, "name", None)
    if hasattr(maybe_name, "name"):
        maybe_name = maybe_name.name
    return maybe_name if isinstance(maybe_name, str) else ""


def _format_attempt_error(exc):
    if isinstance(exc, TimeoutError):
        return f"CollisionError (reported as timeout): {exc}"
    return f"{type(exc).__name__}: {exc}"


def _record_bread_result(results, bread_name, outcome, succeeded_arm, phase, failures):
    results.append(
        {
            "bread_name": bread_name,
            "outcome": outcome,
            "succeeded_arm": succeeded_arm,
            "phase": phase,
            "failure_count": len(failures),
            "failure_reasons": " | ".join(failures),
        }
    )


def _write_results_csv(csv_path, results):
    fieldnames = [
        "bread_name",
        "outcome",
        "succeeded_arm",
        "phase",
        "failure_count",
        "failure_reasons",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def _attach_knives(world):
    knife_right = _parse_stl("pycram_object_gap_demo", "big-knife.stl")
    knife_right.root.name.name = "knife_right"
    knife_left = _parse_stl("pycram_object_gap_demo", "big-knife.stl")
    knife_left.root.name.name = "knife_left"

    l_tip = world.get_body_by_name("l_gripper_tool_frame")
    r_tip = world.get_body_by_name("r_gripper_tool_frame")

    with world.modify_world():
        right_connection = FixedConnection(
            parent=r_tip,
            child=knife_right.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.0,
                y=0.0,
                z=0.08,
                roll=0.0,
                pitch=-np.pi / 2,
                yaw=0.0,
                reference_frame=r_tip,
            ),
        )
        left_connection = FixedConnection(
            parent=l_tip,
            child=knife_left.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.0,
                y=0.0,
                z=-0.08,
                roll=np.pi,
                pitch=np.pi / 2,
                yaw=0.0,
                reference_frame=l_tip,
            ),
        )
        world.merge_world(knife_right, right_connection)
        world.merge_world(knife_left, left_connection)

    right_body = world.get_body_by_name("knife_right")
    left_body = world.get_body_by_name("knife_left")
    return Knife(root=right_body), Knife(root=left_body)


def _collect_breads(world):
    breads = []
    for body in getattr(world, "bodies", []):
        name = _body_name(body)
        if name.startswith("bread_"):
            breads.append(body)
    breads.sort(key=_body_name)
    return breads


def _iter_visual_shapes(body):
    seen = set()
    for owner in (body, getattr(body, "root", None)):
        if owner is None:
            continue
        geom = getattr(owner, "visual", None)
        if geom is None:
            continue
        for shape in getattr(geom, "shapes", []):
            sid = id(shape)
            if sid in seen:
                continue
            seen.add(sid)
            yield shape


def _highlight_current_bread(
    world, breads, active_bread, failed_breads=None, successful_breads=None
):
    failed_breads = failed_breads or set()
    successful_breads = successful_breads or set()
    with world.modify_world():
        for bread in breads:
            if bread is active_bread:
                color = ACTIVE_BREAD_COLOR
            elif bread in failed_breads:
                color = FAILED_BREAD_COLOR
            elif bread in successful_breads:
                color = SUCCESS_BREAD_COLOR
            else:
                color = DEFAULT_BREAD_COLOR
            for shape in _iter_visual_shapes(bread):
                shape.color = color


def _try_cut(context, bread, arm, tool):
    pickup_loc = CostmapLocation(
        target=PoseStamped.from_spatial_type(bread.global_pose),
        reachable_arm=arm,
        reachable_for=context.robot,
    )
    with simulated_robot_without_collision:
        SequentialPlan(
            context,
            ParkArmsActionDescription(Arms.BOTH),
            NavigateActionDescription(pickup_loc, True),
        ).perform()

    with simulated_robot_with_collision:
        current_plan = SequentialPlan(
            context,
            CuttingActionDescription(
                container=bread,
                arm=arm,
                tool=tool,
                technique="saw",
                pointer_stride=13,
                num_cuts_x=1,
            ),
        )
        current_plan.perform()

    dao = to_dao(current_plan)
    session.add(dao)
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"[DB] commit failed: {type(e).__name__}: {e}")


def _rotate_bread_180deg_z(world, bread):
    print("rotating bread")
    pose = bread.global_pose
    pos = np.asarray(pose.to_position().to_np(), dtype=float).reshape(-1)[:3]
    quat = np.asarray(pose.to_quaternion().to_np(), dtype=float).reshape(-1)[:4]
    rot_quat = quaternion_from_euler(0.0, 0.0, np.pi)
    new_quat = quaternion_multiply(rot_quat, quat)

    rotated_pose = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=float(pos[0]),
        pos_y=float(pos[1]),
        pos_z=float(pos[2]),
        quat_x=float(new_quat[0]),
        quat_y=float(new_quat[1]),
        quat_z=float(new_quat[2]),
        quat_w=float(new_quat[3]),
        reference_frame=world.root,
    )
    print(f"rotated object")
    with world.modify_world():
        bread.parent_connection.origin = rotated_pose


def main(seed=None):
    world, _, surface_plan = setup_random_bread_world(seed=seed)

    rclpy.init()
    node = rclpy.create_node("pycram_cut_all_breads_retry")
    tf_wrapper = TFWrapper(node=node)
    TFPublisher(node=node, _world=world)
    VizMarkerPublisher(_world=world, node=node)

    tf_wrapper.wait_for_transform(
        "apartment/apartment_root",
        "pr2/base_footprint",
        timeout=RclpyDuration(seconds=1.0),
        time=Time(),
    )

    right_knife, left_knife = _attach_knives(world)
    breads = _collect_breads(world)

    context = Context.from_world(world)
    context.ros_node = node

    print("[setup] surface plan:")
    for surface_name, area_m2, target_count, placed_count in surface_plan:
        print(
            f"  - {surface_name}: area={area_m2:.3f}m^2 target={target_count} placed={placed_count}"
        )
    print(f"[setup] breads to cut: {len(breads)}")

    success_primary = 0
    success_fallback = 0
    success_rotated_right = 0
    success_rotated_left = 0
    failed = 0
    failed_breads = set()
    successful_breads = set()
    bread_results = []
    _write_results_csv(RESULTS_CSV_PATH, bread_results)

    with simulated_robot_without_collision:
        SequentialPlan(
            context,
            ParkArmsActionDescription(Arms.BOTH),
            MoveTorsoActionDescription(TorsoState.HIGH),
        ).perform()
    for bread in breads:
        attempt_failures = []
        _highlight_current_bread(
            world,
            breads,
            bread,
            failed_breads=failed_breads,
            successful_breads=successful_breads,
        )
        bread_name = _body_name(bread)
        print(f"[cut] {bread_name}: try RIGHT arm")
        try:
            _try_cut(context, bread, Arms.RIGHT, right_knife)
            success_primary += 1
            successful_breads.add(bread)
            _record_bread_result(
                bread_results,
                bread_name,
                "success",
                "RIGHT",
                "primary",
                attempt_failures,
            )
            _write_results_csv(RESULTS_CSV_PATH, bread_results)
            print(f"[ok] {bread_name}: cut with RIGHT arm")
            continue

        except TimeoutError as exc_right_timeout:
            attempt_failures.append(
                f"RIGHT primary -> {_format_attempt_error(exc_right_timeout)}"
            )
            print(
                f"[retry] {bread_name}: RIGHT timed out "
                f"({type(exc_right_timeout).__name__}: {exc_right_timeout})"
            )
        except Exception as exc_right:
            attempt_failures.append(
                f"RIGHT primary -> {_format_attempt_error(exc_right)}"
            )
            print(
                f"[retry] {bread_name}: RIGHT failed "
                f"({type(exc_right).__name__}: {exc_right})"
            )

        print(f"[cut] {bread_name}: try LEFT arm")
        try:
            _try_cut(context, bread, Arms.LEFT, left_knife)
            success_fallback += 1
            successful_breads.add(bread)
            _record_bread_result(
                bread_results,
                bread_name,
                "success",
                "LEFT",
                "fallback",
                attempt_failures,
            )
            _write_results_csv(RESULTS_CSV_PATH, bread_results)
            print(f"[ok] {bread_name}: cut with LEFT arm (fallback)")
            continue
        except TimeoutError as exc_left_timeout:
            attempt_failures.append(
                f"LEFT fallback -> {_format_attempt_error(exc_left_timeout)}"
            )
            print(
                f"[fail] {bread_name}: LEFT timed out "
                f"({type(exc_left_timeout).__name__}: {exc_left_timeout})"
            )
        except Exception as exc_left:
            attempt_failures.append(
                f"LEFT fallback -> {_format_attempt_error(exc_left)}"
            )
            print(
                f"[fail] {bread_name}: LEFT failed "
                f"({type(exc_left).__name__}: {exc_left})"
            )

        print(f"[retry] {bread_name}: rotate 180deg around Z and try again")
        _rotate_bread_180deg_z(world, bread)

        print(f"[cut] {bread_name}: try RIGHT arm after rotation")
        try:
            _try_cut(context, bread, Arms.RIGHT, right_knife)
            success_rotated_right += 1
            successful_breads.add(bread)
            _record_bread_result(
                bread_results,
                bread_name,
                "success",
                "RIGHT",
                "after_rotation",
                attempt_failures,
            )
            _write_results_csv(RESULTS_CSV_PATH, bread_results)
            print(f"[ok] {bread_name}: cut with RIGHT arm after rotation")
            continue
        except TimeoutError as exc_right_rot_timeout:
            attempt_failures.append(
                f"RIGHT after_rotation -> {_format_attempt_error(exc_right_rot_timeout)}"
            )
            print(
                f"[retry] {bread_name}: RIGHT after rotation timed out "
                f"({type(exc_right_rot_timeout).__name__}: {exc_right_rot_timeout})"
            )
        except Exception as exc_right_rot:
            attempt_failures.append(
                f"RIGHT after_rotation -> {_format_attempt_error(exc_right_rot)}"
            )
            print(
                f"[retry] {bread_name}: RIGHT after rotation failed "
                f"({type(exc_right_rot).__name__}: {exc_right_rot})"
            )

        print(f"[cut] {bread_name}: try LEFT arm after rotation")
        try:
            _try_cut(context, bread, Arms.LEFT, left_knife)
            success_rotated_left += 1
            successful_breads.add(bread)
            _record_bread_result(
                bread_results,
                bread_name,
                "success",
                "LEFT",
                "after_rotation",
                attempt_failures,
            )
            _write_results_csv(RESULTS_CSV_PATH, bread_results)
            print(f"[ok] {bread_name}: cut with LEFT arm after rotation")
        except TimeoutError as exc_left_rot_timeout:
            failed += 1
            failed_breads.add(bread)
            attempt_failures.append(
                f"LEFT after_rotation -> {_format_attempt_error(exc_left_rot_timeout)}"
            )
            _record_bread_result(
                bread_results,
                bread_name,
                "failed",
                "",
                "after_rotation",
                attempt_failures,
            )
            _write_results_csv(RESULTS_CSV_PATH, bread_results)
            print(
                f"[fail] {bread_name}: LEFT after rotation timed out "
                f"({type(exc_left_rot_timeout).__name__}: {exc_left_rot_timeout})"
            )
        except Exception as exc_left_rot:
            failed += 1
            failed_breads.add(bread)
            attempt_failures.append(
                f"LEFT after_rotation -> {_format_attempt_error(exc_left_rot)}"
            )
            _record_bread_result(
                bread_results,
                bread_name,
                "failed",
                "",
                "after_rotation",
                attempt_failures,
            )
            _write_results_csv(RESULTS_CSV_PATH, bread_results)
            print(
                f"[fail] {bread_name}: LEFT after rotation failed "
                f"({type(exc_left_rot).__name__}: {exc_left_rot})"
            )

    _highlight_current_bread(
        world,
        breads,
        None,
        failed_breads=failed_breads,
        successful_breads=successful_breads,
    )

    print("[summary]")
    print(f"  total breads: {len(breads)}")
    print(f"  success primary (RIGHT): {success_primary}")
    print(f"  success fallback (LEFT): {success_fallback}")
    print(f"  success after rotation (RIGHT): {success_rotated_right}")
    print(f"  success after rotation (LEFT): {success_rotated_left}")
    print(f"  failed both arms: {failed}")
    print(f"  results csv: {RESULTS_CSV_PATH}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    session = pycram_sessionmaker()()
    drop_database(session.bind)
    Base.metadata.create_all(session.bind)
    session.commit()

    main()
