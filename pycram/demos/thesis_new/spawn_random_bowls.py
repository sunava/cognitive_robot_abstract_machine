import os
import numpy as np

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.pose import PoseStamped
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Color

from demos.thesis_new.spawn_random_breads import (
    _body_basename,
    _body_name,
    _collect_surface_bodies,
    _count_for_surface,
    _is_pose_reachable_for_cutting,
    _pose_xyz,
    _sample_xy,
    _set_uniform_scale,
    _surface_sampling_bounds,
    _surface_usable_area,
    _tint_surfaces_light_brown,
    body_local_aabb,
)
from demos.thesis_new.world_setup import setup_thesis_world

RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "resources")
)
BOWL_RADIUS_SAFETY_FACTOR = 1.08
MIN_BOWL_CLEARANCE_M = 0.03
STRICT_CLEAN_MODE = True
DEFAULT_BOWL_COLOR = Color(R=0.78, G=0.80, B=0.86)
VERTICAL_WIPE_SURFACE_CANDIDATES = (
    ("cabinet3", 2),
    ("cabinet9", 2),
    ("iai_fridge_door", 2),
    ("oven_area_oven_door", 2),
    ("sink_area_dish_washer_door", 2),
)


def _parse_stl(*relative_path_parts):
    return STLParser(os.path.join(RESOURCES_DIR, *relative_path_parts)).parse()


def _bowl_base_xy_radius():
    bowl = _parse_stl("objects", "bowl.stl")
    mins, maxs = body_local_aabb(bowl.root, use_visual=False, apply_shape_scale=True)
    dx = max(0.0, float(maxs[0] - mins[0]))
    dy = max(0.0, float(maxs[1] - mins[1]))
    return 0.5 * np.hypot(dx, dy)


def _spawn_bowl_at_local_pose(
    world, surface_body, bowl_name, scale, x_local, y_local, yaw, z_local
):
    bowl = _parse_stl("objects", "bowl.stl")
    bowl.root.name.name = bowl_name
    _set_uniform_scale(
        bowl,
        (scale, scale, scale),
        color=DEFAULT_BOWL_COLOR,
    )
    local_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=float(x_local),
        y=float(y_local),
        z=float(z_local),
        roll=0.0,
        pitch=0.0,
        yaw=float(yaw),
        reference_frame=surface_body,
    )
    world_pose = world.transform(local_pose, world.root)
    world.merge_world_at_pose(bowl, world_pose)
    return world_pose


def _sample_vertical_wipe_targets(world, rng, surface_name, count, start_idx):
    try:
        surface_body = world.get_body_by_name(surface_name)
    except Exception:
        return [], []

    mins, maxs = body_local_aabb(
        surface_body, use_visual=False, apply_shape_scale=True
    )
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
    return resolved


def _sample_random_bowl_layout(world, seed=None, spawn_bowls=True):
    rng = np.random.default_rng(seed)

    surfaces = _collect_surface_bodies(world)
    if not surfaces:
        raise RuntimeError("No support surfaces found for random bowl placement.")

    scale_choices = np.array([0.8, 1.0, 1.2, 1.4, 1.6], dtype=float)
    base_radius = _bowl_base_xy_radius()
    spawn_context = Context.from_world(world)
    spawn_robot = spawn_context.robot

    placements = []
    surface_plan = []
    created_idx = 0
    with world.modify_world():
        _tint_surfaces_light_brown(world)
        for surface_body in surfaces:
            surface_name = _body_name(surface_body) or "unknown_surface"
            mins, maxs, lo_x, hi_x, lo_y, hi_y = _surface_sampling_bounds(surface_body)
            z_local = float(maxs[2] + 0.05)
            area_m2 = _surface_usable_area(lo_x, hi_x, lo_y, hi_y)
            target_count = _count_for_surface(surface_name, area_m2)
            # target_count = 1
            occupied_xy = []
            for _ in range(target_count):
                scale = float(rng.choice(scale_choices))
                radius = base_radius * scale * BOWL_RADIUS_SAFETY_FACTOR
                placed = False
                for _attempt in range(120):
                    lo_x_eff = lo_x + radius
                    hi_x_eff = hi_x - radius
                    lo_y_eff = lo_y + radius
                    hi_y_eff = hi_y - radius
                    x_local, y_local = _sample_xy(
                        lo_x_eff, hi_x_eff, lo_y_eff, hi_y_eff, mins, maxs, rng
                    )
                    if all(
                        ((x_local - ox) ** 2 + (y_local - oy) ** 2)
                        >= (
                            radius
                            + orad
                            + (MIN_BOWL_CLEARANCE_M if STRICT_CLEAN_MODE else 0.015)
                        )
                        ** 2
                        for ox, oy, orad in occupied_xy
                    ):
                        yaw = float(rng.uniform(-np.pi, np.pi))
                        candidate_local_pose = (
                            HomogeneousTransformationMatrix.from_xyz_rpy(
                                x=float(x_local),
                                y=float(y_local),
                                z=float(z_local),
                                roll=0.0,
                                pitch=0.0,
                                yaw=float(yaw),
                                reference_frame=surface_body,
                            )
                        )
                        candidate_world_pose = world.transform(
                            candidate_local_pose, world.root
                        )
                        target_pose = PoseStamped.from_spatial_type(
                            candidate_world_pose
                        )
                        if not _is_pose_reachable_for_cutting(
                            spawn_robot, world, target_pose
                        ):
                            continue

                        created_idx += 1
                        bowl_name = f"bowl_{created_idx:04d}"
                        world_pose = candidate_world_pose
                        if spawn_bowls:
                            world_pose = _spawn_bowl_at_local_pose(
                                world=world,
                                surface_body=surface_body,
                                bowl_name=bowl_name,
                                scale=scale,
                                x_local=x_local,
                                y_local=y_local,
                                yaw=yaw,
                                z_local=z_local,
                            )
                        occupied_xy.append((x_local, y_local, radius))
                        placements.append(
                            {
                                "bowl_name": bowl_name,
                                "surface_name": surface_name,
                                "scale": scale,
                                "pose_xyz": _pose_xyz(world_pose),
                                "world_pose": world_pose,
                            }
                        )
                        placed = True
                        break
                if not placed:
                    continue
            surface_plan.append((surface_name, area_m2, target_count, len(occupied_xy)))

    return world, placements, surface_plan


def setup_random_bowl_world(seed=None, robot_name=None, environment_name=None):
    world, placements, surface_plan = _sample_random_bowl_layout(
        setup_thesis_world(robot_name=robot_name, environment_name=environment_name),
        seed=seed,
        spawn_bowls=True,
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
            world, rng, surface_name, count, start_idx=next_idx
        )
        extra_targets.extend(targets)
        extra_surface_plan.extend(plan_entries)
        next_idx += len(targets)

    return world, renamed_placements + extra_targets, surface_plan + extra_surface_plan
