import rclpy

from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time
from pycram.datastructures.enums import Arms
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.robot_mixins import SpecifiesLeftRightArm
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection

from demos.thesis_new.utils.experiment_logging import body_name
from demos.thesis_new.world_setup import resolve_robot_name_from_annotation


def setup_experiment_runtime(world, node_name):
    rclpy.init()
    node = rclpy.create_node(node_name)
    tf_wrapper = TFWrapper(node=node)
    TFPublisher(node=node, _world=world)
    VizMarkerPublisher(_world=world, node=node)
    robot = get_primary_robot(world)
    tf_wrapper.wait_for_transform(
        "apartment/apartment_root",
        str(robot.root.name),
        timeout=RclpyDuration(seconds=1.0),
        time=Time(),
    )
    return node


def shutdown_experiment_runtime(node):
    node.destroy_node()
    rclpy.shutdown()


def collect_named_targets(world, prefix):
    targets = []
    for body in getattr(world, "bodies", []):
        name = body_name(body)
        if name.startswith(prefix):
            targets.append(body)
    targets.sort(key=body_name)
    return targets


def iter_visual_shapes(body):
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


def highlight_current_target(
    world,
    targets,
    active_target,
    *,
    default_color,
    active_color,
    failed_color,
    success_color,
    failed_targets=None,
    successful_targets=None,
):
    failed_targets = failed_targets or set()
    successful_targets = successful_targets or set()
    with world.modify_world():
        for target in targets:
            if target is active_target:
                color = active_color
            elif target in failed_targets:
                color = failed_color
            elif target in successful_targets:
                color = success_color
            else:
                color = default_color
            for shape in iter_visual_shapes(target):
                shape.color = color


def attach_bimanual_tools(
    world,
    parse_stl_fn,
    *,
    mesh_parts,
    right_name,
    left_name,
    right_pose_kwargs,
    left_pose_kwargs,
    tool_cls,
):
    right_tool = parse_stl_fn(*mesh_parts)
    right_tool.root.name.name = right_name
    left_tool = parse_stl_fn(*mesh_parts)
    left_tool.root.name.name = left_name

    l_tip, r_tip = get_bimanual_tool_frames(world)

    with world.modify_world():
        right_connection = FixedConnection(
            parent=r_tip,
            child=right_tool.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                reference_frame=r_tip, **right_pose_kwargs
            ),
        )
        left_connection = FixedConnection(
            parent=l_tip,
            child=left_tool.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                reference_frame=l_tip, **left_pose_kwargs
            ),
        )
        world.merge_world(right_tool, right_connection)
        world.merge_world(left_tool, left_connection)

    right_body = world.get_body_by_name(right_name)
    left_body = world.get_body_by_name(left_name)
    return tool_cls(root=right_body), tool_cls(root=left_body)


def get_available_arm_tool_frames(world):
    robot = get_primary_robot(world)
    if not getattr(robot, "arms", None):
        raise ValueError(f"Robot '{body_name(robot.root)}' does not expose any arms.")

    if isinstance(robot, SpecifiesLeftRightArm) and len(robot.arms) == 2:
        return [
            (Arms.RIGHT, robot.right_arm.manipulator.tool_frame),
            (Arms.LEFT, robot.left_arm.manipulator.tool_frame),
        ]

    if len(robot.arms) == 1:
        return [(Arms.LEFT, robot.arms[0].manipulator.tool_frame)]

    raise ValueError(
        f"Robot '{body_name(robot.root)}' exposes unsupported arm configuration with {len(robot.arms)} arms."
    )


def get_available_arms(world):
    return [arm for arm, _ in get_available_arm_tool_frames(world)]


def get_park_arms_argument(world):
    available_arms = get_available_arms(world)
    if len(available_arms) == 2:
        return Arms.BOTH
    return available_arms[0]


def attach_available_tools(
    world,
    parse_stl_fn,
    *,
    mesh_parts,
    right_name,
    left_name,
    right_pose_kwargs,
    left_pose_kwargs,
    tool_cls,
):
    available_frames = get_available_arm_tool_frames(world)
    tool_by_arm = {}

    with world.modify_world():
        for arm, tip in available_frames:
            if arm == Arms.RIGHT:
                tool_world = parse_stl_fn(*mesh_parts)
                tool_world.root.name.name = right_name
                tool_name = right_name
                pose_kwargs = right_pose_kwargs
            else:
                tool_world = parse_stl_fn(*mesh_parts)
                tool_world.root.name.name = left_name
                tool_name = left_name
                pose_kwargs = left_pose_kwargs

            connection = FixedConnection(
                parent=tip,
                child=tool_world.root,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    reference_frame=tip, **pose_kwargs
                ),
            )
            world.merge_world(tool_world, connection)
            tool_by_arm[arm] = tool_cls(root=connection.child)

    return [(arm, tool_by_arm[arm]) for arm, _ in available_frames]


def get_primary_robot(world):
    robots = world.get_semantic_annotations_by_type(AbstractRobot)
    if not robots:
        raise RuntimeError("No semantic robot found in world.")
    return robots[0]


def get_primary_robot_name(world):
    return resolve_robot_name_from_annotation(get_primary_robot(world))


def get_bimanual_tool_frames(world):
    robot = get_primary_robot(world)
    if not isinstance(robot, SpecifiesLeftRightArm):
        raise ValueError(
            f"Robot '{body_name(robot.root)}' does not expose distinct left/right arms."
        )
    if len(robot.arms) != 2:
        raise ValueError(
            f"Robot '{body_name(robot.root)}' requires exactly two arms for this demo, found {len(robot.arms)}."
        )
    return robot.left_arm.manipulator.tool_frame, robot.right_arm.manipulator.tool_frame


def commit_plan_to_db(session, current_plan):
    from krrood.ormatic.dao import to_dao

    dao = to_dao(current_plan)
    session.add(dao)
    try:
        session.commit()
    except Exception as exc:
        session.rollback()
        print(f"[DB] commit failed: {type(exc).__name__}: {exc}")
