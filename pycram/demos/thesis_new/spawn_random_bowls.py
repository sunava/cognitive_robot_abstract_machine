import os
import numpy as np

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Color

from demos.thesis_new.spawn_random_breads import (
    _body_basename,
    _body_name,
    _is_excluded_kitchen_spawn_pose,
    _pose_xyz,
    _sample_random_surface_layout,
    body_local_aabb,
)
from demos.thesis_new.world_setup import setup_thesis_world
from demos.thesis_new.utils.demo_utils import build_navigation_costmaps

RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "resources")
)
BOWL_RADIUS_SAFETY_FACTOR = 1.08
MIN_BOWL_CLEARANCE_M = 0.03
STRICT_CLEAN_MODE = True
DEFAULT_BOWL_COLOR = Color(R=0.78, G=0.80, B=0.86)
DEFAULT_POT_COLOR = Color(R=0.56, G=0.58, B=0.62)
VERTICAL_WIPE_SURFACE_CANDIDATES = (
    ("cabinet3", 2),
    ("cabinet9", 2),
    ("iai_fridge_door", 2),
    ("oven_area_oven_door", 2),
    ("sink_area_dish_washer_door", 2),
)
GENERIC_VERTICAL_WIPE_KEYWORDS = (
    "cabinet",
    "shelf",
    # "wall",
    "door",
)
GENERIC_VERTICAL_WIPE_EXCLUDE_KEYWORDS = (
    "chair",
    "table",
    "counter",
    "sofa",
    "plant",
    "lamp",
    "marker",
    "cone",
    "hydrant",
    "tree",
    "human",
)


def _parse_stl(*relative_path_parts):
    return STLParser(os.path.join(RESOURCES_DIR, *relative_path_parts)).parse()


def _container_base_xy_radius(*mesh_parts):
    bowl = _parse_stl(*mesh_parts)
    mins, maxs = body_local_aabb(bowl.root, use_visual=False, apply_shape_scale=True)
    dx = max(0.0, float(maxs[0] - mins[0]))
    dy = max(0.0, float(maxs[1] - mins[1]))
    return 0.5 * np.hypot(dx, dy)


def _bowl_base_xy_radius():
    return _container_base_xy_radius("objects", "bowl.stl")


def _sample_vertical_wipe_targets(
    world, rng, surface_name, count, start_idx, *, environment_name=None
):
    try:
        surface_body = world.get_body_by_name(surface_name)
    except Exception:
        return [], []

    mins, maxs = body_local_aabb(surface_body, use_visual=False, apply_shape_scale=True)
    extents = maxs - mins
    y_margin = min(0.08, max(0.0, 0.18 * extents[1]))
    z_margin = min(0.10, max(0.0, 0.18 * extents[2]))
    lo_y = mins[1] + y_margin
    hi_y = maxs[1] - y_margin
    lo_z = mins[2] + z_margin
    hi_z = maxs[2] - z_margin

    placements = []
    occupied_yz = []
    door_offset = 0.03
    yz_clearance = 0.18

    for local_idx in range(count):
        for _attempt in range(80):
            if hi_y <= lo_y:
                y_local = 0.5 * (mins[1] + maxs[1])
            else:
                y_local = float(rng.uniform(lo_y, hi_y))
            if hi_z <= lo_z:
                z_local = 0.5 * (mins[2] + maxs[2])
            else:
                z_local = float(rng.uniform(lo_z, hi_z))

            if any(
                ((y_local - oy) ** 2 + (z_local - oz) ** 2) < yz_clearance**2
                for oy, oz in occupied_yz
            ):
                continue

            local_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=float(maxs[0] + door_offset),
                y=float(y_local),
                z=float(z_local),
                roll=0.0,
                pitch=np.pi / 2,
                yaw=0.0,
                reference_frame=surface_body,
            )
            world_pose = world.transform(local_pose, world.root)
            if _is_excluded_kitchen_spawn_pose(
                world_pose, environment_name=environment_name
            ):
                continue
            placements.append(
                {
                    "bowl_name": f"wipe_target_{start_idx + local_idx:04d}",
                    "surface_name": str(surface_body.name),
                    "scale": 1.0,
                    "pose_xyz": _pose_xyz(world_pose),
                    "world_pose": world_pose,
                }
            )
            occupied_yz.append((y_local, z_local))
            break

    surface_area = max(0.0, float(hi_y - lo_y)) * max(0.0, float(hi_z - lo_z))
    surface_plan_entry = (
        str(surface_body.name),
        surface_area,
        count,
        len(placements),
    )
    return placements, [surface_plan_entry]


def _is_pose_reachable_for_mixing(robot, world, target_pose, cache=None):
    key = None
    if cache is not None:
        position = target_pose.to_position()
        key = (
            int(round(float(position.x) / 0.10)),
            int(round(float(position.y) / 0.10)),
            str(target_pose.reference_frame),
        )
        if key in cache:
            return cache[key]

    try:
        _, _, final_map = build_navigation_costmaps(
            robot,
            world,
            target_pose,
            width=60,
            height=60,
            resolution=0.05,
            ring_std=8,
            ring_distance=0.55,
            obstacle_clearance=0.20,
            number_of_samples=8,
        )
        next(iter(final_map))
        result = True
    except StopIteration:
        result = False
    except Exception:
        result = True

    if cache is not None and key is not None:
        cache[key] = result
    return result


def _resolve_vertical_wipe_surfaces(world):
    resolved = []
    seen = set()
    for candidate_name, count in VERTICAL_WIPE_SURFACE_CANDIDATES:
        for body in getattr(world, "bodies", []):
            body_name = _body_name(body)
            body_basename = _body_basename(body)
            if not body_name or not body_basename:
                continue
            if candidate_name not in (body_name, body_basename):
                continue
            if body_name in seen:
                continue
            resolved.append((body_name, count))
            seen.add(body_name)
            break

    for body in getattr(world, "bodies", []):
        body_name = _body_name(body)
        body_basename = (_body_basename(body) or "").lower()
        if not body_name or not body_basename or body_name in seen:
            continue
        if any(
            skip in body_basename for skip in GENERIC_VERTICAL_WIPE_EXCLUDE_KEYWORDS
        ):
            continue
        if not any(
            keyword in body_basename for keyword in GENERIC_VERTICAL_WIPE_KEYWORDS
        ):
            continue
        resolved.append((body_name, 2))
        seen.add(body_name)
    return resolved


def _sample_random_bowl_layout(
    world,
    seed=None,
    robot_name=None,
    environment_name=None,
    spawn_bowls=True,
    container_kind="bowl",
):
    mesh_parts = (
        ("pycram_object_gap_demo", "pot_1.stl")
        if str(container_kind).lower() == "pot"
        else ("objects", "bowl.stl")
    )
    scale_choices = (
        np.array([0.8], dtype=float)
        if str(container_kind).lower() == "pot"
        else np.array([0.8, 1.0, 1.2, 1.4, 1.6], dtype=float)
    )
    object_color = (
        DEFAULT_POT_COLOR
        if str(container_kind).lower() == "pot"
        else DEFAULT_BOWL_COLOR
    )
    world, tuple_placements, surface_plan = _sample_random_surface_layout(
        world=world,
        seed=seed,
        environment_name=environment_name,
        object_label="bowl",
        object_name_prefix="bowl",
        mesh_parts=mesh_parts,
        object_color=object_color,
        scale_choices=scale_choices,
        base_radius=_container_base_xy_radius(*mesh_parts),
        radius_safety_factor=BOWL_RADIUS_SAFETY_FACTOR,
        min_clearance_m=MIN_BOWL_CLEARANCE_M,
        strict_clean_mode=STRICT_CLEAN_MODE,
        z_offset=0.05,
        reachability_fn=_is_pose_reachable_for_mixing,
        debug_disable_reachability=False,
        debug_force_spawn_all_targets=False,
        debug_spawn_actual_poses=False,
        spawn_objects=spawn_bowls,
        include_world_pose=not spawn_bowls,
    )
    placements = []
    for placement in tuple_placements:
        bowl_name, surface_name, scale, pose_xyz = placement[:4]
        world_pose = placement[4] if len(placement) > 4 else None
        placements.append(
            {
                "bowl_name": bowl_name,
                "surface_name": surface_name,
                "scale": scale,
                "pose_xyz": pose_xyz,
                "world_pose": world_pose,
            }
        )
    return world, placements, surface_plan


def setup_random_bowl_world(seed=None, robot_name=None, environment_name=None):
    return setup_random_mixing_container_world(
        seed=seed,
        robot_name=robot_name,
        environment_name=environment_name,
        container_kind="bowl",
    )


def setup_random_mixing_container_world(
    seed=None, robot_name=None, environment_name=None, container_kind="bowl"
):
    world, placements, surface_plan = _sample_random_bowl_layout(
        setup_thesis_world(robot_name=robot_name, environment_name=environment_name),
        seed=seed,
        robot_name=robot_name,
        environment_name=environment_name,
        spawn_bowls=True,
        container_kind=container_kind,
    )
    return (
        world,
        [
            (
                placement["bowl_name"],
                placement["surface_name"],
                placement["scale"],
                placement["pose_xyz"],
            )
            for placement in placements
        ],
        surface_plan,
    )


def sample_random_bowl_poses(seed=None, robot_name=None, environment_name=None):
    world, placements, surface_plan = _sample_random_bowl_layout(
        setup_thesis_world(robot_name=robot_name, environment_name=environment_name),
        seed=seed,
        robot_name=robot_name,
        environment_name=environment_name,
        spawn_bowls=False,
    )
    rng = np.random.default_rng(seed)

    renamed_placements = []
    for idx, placement in enumerate(placements, start=1):
        renamed = dict(placement)
        renamed["bowl_name"] = f"wipe_target_{idx:04d}"
        renamed_placements.append(renamed)

    extra_targets = []
    extra_surface_plan = []
    next_idx = len(renamed_placements) + 1
    for surface_name, count in _resolve_vertical_wipe_surfaces(world):
        targets, plan_entries = _sample_vertical_wipe_targets(
            world,
            rng,
            surface_name,
            count,
            start_idx=next_idx,
            environment_name=environment_name,
        )
        extra_targets.extend(targets)
        extra_surface_plan.extend(plan_entries)
        next_idx += len(targets)

    return world, renamed_placements + extra_targets, surface_plan + extra_surface_plan
