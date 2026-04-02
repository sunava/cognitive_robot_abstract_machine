from __future__ import annotations

from typing import Any, Dict

import numpy as np
from trimesh.proximity import closest_point

from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from .world_utils import body_local_aabb


def points_world_to_body(
    points_world: np.ndarray,
    world: World,
    body: Body,
) -> np.ndarray:
    """Transform Nx3 points from world frame to the given body frame."""
    P = np.asarray(points_world, dtype=float).reshape(-1, 3)
    points_body = np.empty_like(P, dtype=float)

    for i, p in enumerate(P):
        p_world = Point3(
            x=float(p[0]),
            y=float(p[1]),
            z=float(p[2]),
            reference_frame=world.root,
        )
        p_body = world.transform(p_world, body)
        points_body[i] = np.asarray(p_body.to_np()[:3], dtype=float)

    return points_body


def distance_to_mesh_metrics(
    points_world: np.ndarray,
    world: World,
    body: Body,
    threshold_m: float = 0.005,
) -> Dict[str, Any]:
    """
    Compute distance-to-mesh metrics for points against body's collision mesh.
    """
    points_body = points_world_to_body(points_world, world, body)
    mesh = body.collision[0].mesh
    _, dists, face_ids = closest_point(mesh, points_body)
    dists = np.asarray(dists, dtype=float)

    if dists.size == 0:
        return {
            "num_points": 0,
            "min_distance": float("inf"),
            "mean_distance": float("inf"),
            "below_threshold_ratio": 0.0,
            "closest_face_id": None,
            "distance_success": False,
        }

    min_idx = int(np.argmin(dists))
    below_ratio = float(np.mean(dists <= float(threshold_m)))
    return {
        "num_points": int(dists.size),
        "min_distance": float(np.min(dists)),
        "mean_distance": float(np.mean(dists)),
        "below_threshold_ratio": below_ratio,
        "closest_face_id": int(face_ids[min_idx]),
        "distance_success": below_ratio > 0.5,
    }


def cutting_depth_metrics(
    points_world: np.ndarray,
    world: World,
    bread_body: Body,
    apply_shape_scale: bool = True,
) -> Dict[str, Any]:
    """
    Check if trajectory crosses from above top surface to below cut depth inside bread XY bounds.
    """
    points_body = points_world_to_body(points_world, world, bread_body)
    mins, maxs = body_local_aabb(
        bread_body, use_visual=False, apply_shape_scale=apply_shape_scale
    )

    size_z = maxs[2] - mins[2]
    z_top = float(maxs[2])
    z_cut = float(mins[2] + max(0.003, 0.05 * size_z))

    inside_xy = (
        (points_body[:, 0] >= mins[0])
        & (points_body[:, 0] <= maxs[0])
        & (points_body[:, 1] >= mins[1])
        & (points_body[:, 1] <= maxs[1])
    )
    above_top = points_body[:, 2] > z_top

    has_entry = bool(np.any(inside_xy & above_top))

    return {
        "z_top": z_top,
        "z_cut": z_cut,
        "inside_xy_ratio": float(np.mean(inside_xy)),
        "has_entry_from_above": has_entry,
    }


def mixing_bowl_metrics(
    points_world: np.ndarray,
    world: World,
    bowl_body: Body,
    apply_shape_scale: bool = True,
    margin: float = 0.005,
    inner_band_min_ratio: float = 0.6,
) -> Dict[str, Any]:
    """
    Approximate bowl quality:
    - points inside a cylindrical bowl proxy
    - points near the interior wall band
    """
    points_body = points_world_to_body(points_world, world, bowl_body)
    mins, maxs = body_local_aabb(
        bowl_body, use_visual=False, apply_shape_scale=apply_shape_scale
    )

    cx = 0.5 * (mins[0] + maxs[0])
    cy = 0.5 * (mins[1] + maxs[1])
    radius = max(1e-9, 0.5 * min(maxs[0] - mins[0], maxs[1] - mins[1]) - margin)
    z_min = float(mins[2] + margin)
    z_max = float(maxs[2] - margin)

    dxy = points_body[:, :2] - np.array([cx, cy], dtype=float)
    r = np.linalg.norm(dxy, axis=1)
    in_r = r <= radius
    in_z = (points_body[:, 2] >= z_min) & (points_body[:, 2] <= z_max)
    inside = in_r & in_z
    near_interior = inside & (r >= inner_band_min_ratio * radius)

    inside_ratio = float(np.mean(inside))
    near_ratio = float(np.mean(near_interior))
    return {
        "inside_ratio": inside_ratio,
        "near_interior_ratio": near_ratio,
        "radius_proxy": float(radius),
        "z_min_proxy": z_min,
        "z_max_proxy": z_max,
        "mixing_success": inside_ratio > 0.8 and near_ratio > 0.3,
    }
