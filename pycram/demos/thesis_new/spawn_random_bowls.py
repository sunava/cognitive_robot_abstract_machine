import os
import numpy as np

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.pose import PoseStamped
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Color

from demos.thesis_new.spawn_random_breads import (
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
from pycram.testing import setup_world

RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "resources")
)
BOWL_RADIUS_SAFETY_FACTOR = 1.08
MIN_BOWL_CLEARANCE_M = 0.03
STRICT_CLEAN_MODE = True
DEFAULT_BOWL_COLOR = Color(R=0.78, G=0.80, B=0.86)


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


def setup_random_bowl_world(seed=None):
    world = setup_world()
    rng = np.random.default_rng(seed)

    surfaces = _collect_surface_bodies(world)
    if not surfaces:
        raise RuntimeError("No apartment surfaces found for random bowl placement.")

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
                            (bowl_name, surface_name, scale, _pose_xyz(world_pose))
                        )
                        placed = True
                        break
                if not placed:
                    continue
            surface_plan.append((surface_name, area_m2, target_count, len(occupied_xy)))

    return world, placements, surface_plan
