import rclpy

from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection

from demos.thesis_new.utils.experiment_logging import body_name


def setup_experiment_runtime(world, node_name):
    rclpy.init()
    node = rclpy.create_node(node_name)
    tf_wrapper = TFWrapper(node=node)
    TFPublisher(node=node, _world=world)
    VizMarkerPublisher(_world=world, node=node)
    tf_wrapper.wait_for_transform(
        "apartment/apartment_root",
        "pr2/base_footprint",
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

    l_tip = world.get_body_by_name("l_gripper_tool_frame")
    r_tip = world.get_body_by_name("r_gripper_tool_frame")

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


def commit_plan_to_db(session, current_plan):
    from krrood.ormatic.dao import to_dao

    dao = to_dao(current_plan)
    session.add(dao)
    try:
        session.commit()
    except Exception as exc:
        session.rollback()
        print(f"[DB] commit failed: {type(exc).__name__}: {exc}")
