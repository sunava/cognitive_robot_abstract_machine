import os
import numpy as np
import rclpy

from pycram.testing import setup_world
from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Color, Scale

from demos.thesis_new.thesis_math.world_utils import body_local_aabb

RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "resources")
)

PREFERRED_SURFACE_NAMES = (
    "island_countertop",
    "countertop",
    "table_area_main",
    "coffee_table",
    "bedside_table",
)

# Automatic count model: breads ~= usable_surface_area * BREADS_PER_SQM.
# Keep this tunable when switching to a new environment.
BREADS_PER_SQM = 12.0
MIN_BREADS_PER_SURFACE = 0
MAX_BREADS_PER_SURFACE = 30
STRICT_CLEAN_MODE = True
MIN_BREAD_CLEARANCE_M = 0.03
BREAD_RADIUS_SAFETY_FACTOR = 1.08

# Optional per-surface overrides (set value to force exact count).
SURFACE_COUNT_OVERRIDES = {}
COUNTERTOP_TINT = Color(R=0.82, G=0.70, B=0.55)


def _parse_stl(*relative_path_parts):
    return STLParser(os.path.join(RESOURCES_DIR, *relative_path_parts)).parse()


def _set_uniform_scale(body, scale_xyz, color=None):
    scale = Scale(*scale_xyz)
    for shape in body.root.visual.shapes:
        shape.scale = scale
        if color is not None:
            shape.color = color
    for shape in body.root.collision.shapes:
        shape.scale = scale


def _body_name(body):
    maybe_name = getattr(body, "name", None)
    if hasattr(maybe_name, "name"):
        maybe_name = maybe_name.name
    return maybe_name if isinstance(maybe_name, str) else None


def _surface_like_name(name):
    lname = name.lower()
    if ("counter" not in lname) and ("table" not in lname):
        return False
    if any(skip in lname for skip in ("drawer", "door", "handle", "waterfall", "back")):
        return False
    return True


def _collect_surface_bodies(world):
    surfaces = []
    seen = set()

    for name in PREFERRED_SURFACE_NAMES:
        try:
            body = world.get_body_by_name(name)
        except Exception:
            continue
        surfaces.append(body)
        seen.add(name)

    for body in getattr(world, "bodies", []):
        name = _body_name(body)
        if not name or name in seen:
            continue

        if not _surface_like_name(name):
            continue
        surfaces.append(body)
        seen.add(name)

    return surfaces


def _iter_visual_shapes(body):
    seen = set()
    for geom_owner in (body, getattr(body, "root", None)):
        if geom_owner is None:
            continue
        geom = getattr(geom_owner, "visual", None)
        if geom is None:
            continue
        for shape in getattr(geom, "shapes", []):
            sid = id(shape)
            if sid in seen:
                continue
            seen.add(sid)
            yield shape


def _tint_surfaces_light_brown(world):
    for body in getattr(world, "bodies", []):
        name = _body_name(body) or ""
        if not _surface_like_name(name):
            continue
        for shape in _iter_visual_shapes(body):
            shape.color = COUNTERTOP_TINT


def _surface_sampling_bounds(surface_body):
    mins, maxs = body_local_aabb(
        surface_body, use_visual=False, apply_shape_scale=True
    )
    extents = maxs - mins

    margin_x = min(0.08, max(0.0, 0.25 * extents[0]))
    margin_y = min(0.08, max(0.0, 0.25 * extents[1]))

    lo_x = mins[0] + margin_x
    hi_x = maxs[0] - margin_x
    lo_y = mins[1] + margin_y
    hi_y = maxs[1] - margin_y

    return mins, maxs, lo_x, hi_x, lo_y, hi_y


def _sample_xy(lo_x, hi_x, lo_y, hi_y, mins, maxs, rng):
    if hi_x <= lo_x:
        x_local = 0.5 * (mins[0] + maxs[0])
    else:
        x_local = float(rng.uniform(lo_x, hi_x))
    if hi_y <= lo_y:
        y_local = 0.5 * (mins[1] + maxs[1])
    else:
        y_local = float(rng.uniform(lo_y, hi_y))
    return x_local, y_local


def _bread_base_xy_radius():
    """Approximate bread XY footprint as a circle radius in local bread frame."""
    bread = _parse_stl("pycram_object_gap_demo", "bread.stl")
    mins, maxs = body_local_aabb(
        bread.root, use_visual=False, apply_shape_scale=True
    )
    dx = max(0.0, float(maxs[0] - mins[0]))
    dy = max(0.0, float(maxs[1] - mins[1]))
    return 0.5 * np.hypot(dx, dy)


def _surface_usable_area(lo_x, hi_x, lo_y, hi_y):
    width = max(0.0, float(hi_x - lo_x))
    depth = max(0.0, float(hi_y - lo_y))
    return width * depth


def _count_for_surface(surface_name, area_m2):
    if surface_name in SURFACE_COUNT_OVERRIDES:
        return int(SURFACE_COUNT_OVERRIDES[surface_name])
    raw = int(round(area_m2 * BREADS_PER_SQM))
    return max(MIN_BREADS_PER_SURFACE, min(MAX_BREADS_PER_SURFACE, raw))


def _pose_xyz(spatial_pose):
    position = spatial_pose.to_position()
    if hasattr(position, "is_constant") and position.is_constant():
        xyz = position.to_np()
    elif hasattr(position, "evaluate"):
        xyz = np.asarray(position.evaluate(), dtype=float).reshape(-1)
    else:
        xyz = np.asarray(position.to_np(), dtype=float).reshape(-1)
    return float(xyz[0]), float(xyz[1]), float(xyz[2])


def _spawn_bread_at_local_pose(world, surface_body, bread_name, scale, x_local, y_local, yaw, z_local):
    bread = _parse_stl("pycram_object_gap_demo", "bread.stl")
    bread.root.name.name = bread_name
    _set_uniform_scale(
        bread,
        (scale, scale, scale),
        color=Color(R=0.76, G=0.60, B=0.42),
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
    world.merge_world_at_pose(bread, world_pose)
    return world_pose


def setup_random_bread_world(seed=None):
    world = setup_world()
    rng = np.random.default_rng(seed)

    surfaces = _collect_surface_bodies(world)
    if not surfaces:
        raise RuntimeError("No apartment surfaces found for random bread placement.")

    scale_choices = np.array([0.8, 1.0, 1.2, 1.4, 1.6], dtype=float)
    base_radius = _bread_base_xy_radius()

    placements = []
    surface_plan = []
    created_idx = 0
    with world.modify_world():
        _tint_surfaces_light_brown(world)
        for surface_body in surfaces:
            surface_name = _body_name(surface_body) or "unknown_surface"
            mins, maxs, lo_x, hi_x, lo_y, hi_y = _surface_sampling_bounds(surface_body)
            z_local = float(maxs[2] + 0.02)
            area_m2 = _surface_usable_area(lo_x, hi_x, lo_y, hi_y)
            target_count = _count_for_surface(surface_name, area_m2)
            occupied_xy = []
            for _ in range(target_count):
                scale = float(rng.choice(scale_choices))
                # Scale-aware safety radius in local surface XY for clean spacing.
                radius = base_radius * scale * BREAD_RADIUS_SAFETY_FACTOR
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
                            + (MIN_BREAD_CLEARANCE_M if STRICT_CLEAN_MODE else 0.015)
                        )
                        ** 2
                        for ox, oy, orad in occupied_xy
                    ):
                        yaw = float(rng.uniform(-np.pi, np.pi))
                        created_idx += 1
                        bread_name = f"bread_{created_idx:04d}"
                        world_pose = _spawn_bread_at_local_pose(
                            world=world,
                            surface_body=surface_body,
                            bread_name=bread_name,
                            scale=scale,
                            x_local=x_local,
                            y_local=y_local,
                            yaw=yaw,
                            z_local=z_local,
                        )
                        occupied_xy.append((x_local, y_local, radius))
                        placements.append(
                            (bread_name, surface_name, scale, _pose_xyz(world_pose))
                        )
                        placed = True
                        break
                if not placed:
                    continue
            surface_plan.append((surface_name, area_m2, target_count, len(occupied_xy)))

    return world, placements, surface_plan


def main(seed=None):
    world, placements, surface_plan = setup_random_bread_world(seed=seed)

    rclpy.init()
    node = rclpy.create_node("pycram_random_breads_demo")
    tf_wrapper = TFWrapper(node=node)
    TFPublisher(node=node, _world=world)
    VizMarkerPublisher(_world=world, node=node)

    tf_wrapper.wait_for_transform(
        "apartment/apartment_root",
        "pr2/base_footprint",
        timeout=RclpyDuration(seconds=1.0),
        time=Time(),
    )

    for surface_name, area_m2, target_count, placed_count in surface_plan:
        print(
            f"[surface] {surface_name}: usable_area={area_m2:.3f} m^2 "
            f"target={target_count} placed={placed_count}"
        )

    for bread_name, surface_name, scale, (x, y, z) in placements:
        print(
            f"{bread_name} spawned on {surface_name} at "
            f"({x:.3f}, {y:.3f}, {z:.3f}) scale={scale:.2f}"
        )

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
