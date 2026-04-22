import numpy as np

from pycram.robot_plans.actions.composite.thesis_math.motion_models import (
    MotionSegment,
    MotionSequence,
)
from pycram.robot_plans.actions.composite.thesis_math.motion_profiles import (
    ShearProfile,
    ShearXYProfile,
    clamp_to_cylinder_xy,
    fixed_rpy,
    make_constrained_curve,
    oscillatory_shear_local_profiled,
    oscillatory_shear_xy_profiled,
    planar_raster_xy,
    planar_spiral_xy,
    planar_sweep_x,
    rot_y,
)
from pycram.robot_plans.actions.composite.thesis_math.world_utils import body_local_aabb

APPROACH_Z_EXTRA_CLEARANCE_M = 0.02


def _cross_2d(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull_xy(points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    if len(pts) <= 1:
        return pts
    pts = np.unique(np.round(pts, decimals=8), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _point_in_convex_polygon_xy(point_xy: np.ndarray, hull_xy: np.ndarray) -> bool:
    hull = np.asarray(hull_xy, dtype=float).reshape(-1, 2)
    point = np.asarray(point_xy, dtype=float).reshape(2)
    if len(hull) < 3:
        return True
    eps = 1e-9
    prev_sign = 0.0
    for i in range(len(hull)):
        a = hull[i]
        b = hull[(i + 1) % len(hull)]
        cross = _cross_2d(a, b, point)
        if abs(cross) <= eps:
            continue
        sign = np.sign(cross)
        if prev_sign == 0.0:
            prev_sign = sign
        elif sign != prev_sign:
            return False
    return True


def _project_point_to_segment_xy(
    point_xy: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray
) -> np.ndarray:
    p = np.asarray(point_xy, dtype=float).reshape(2)
    a = np.asarray(a_xy, dtype=float).reshape(2)
    b = np.asarray(b_xy, dtype=float).reshape(2)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return a.copy()
    t = float(np.dot(p - a, ab) / denom)
    t = np.clip(t, 0.0, 1.0)
    return a + t * ab


def _constrain_to_convex_hull_xy(
    q_local: np.ndarray, hull_xy: np.ndarray
) -> np.ndarray:
    q = np.asarray(q_local, dtype=float).reshape(3)
    if len(hull_xy) < 3 or _point_in_convex_polygon_xy(q[:2], hull_xy):
        return q

    best_xy = None
    best_dist = float("inf")
    for i in range(len(hull_xy)):
        a = hull_xy[i]
        b = hull_xy[(i + 1) % len(hull_xy)]
        candidate_xy = _project_point_to_segment_xy(q[:2], a, b)
        dist = float(np.linalg.norm(q[:2] - candidate_xy))
        if dist < best_dist:
            best_dist = dist
            best_xy = candidate_xy

    return np.array([best_xy[0], best_xy[1], q[2]], dtype=float)


def _top_surface_hull_xy(surface_body, upward_threshold=0.6):
    mesh = getattr(surface_body, "combined_mesh", None)
    if mesh is None or getattr(mesh, "is_empty", True):
        return None

    normals = np.asarray(mesh.face_normals, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(normals) == 0 or len(faces) == 0 or len(vertices) == 0:
        return None

    upward_mask = normals[:, 2] > float(upward_threshold)
    if not upward_mask.any():
        return None

    upward_faces = faces[upward_mask]
    top_vertices = vertices[np.unique(upward_faces.reshape(-1))]
    if len(top_vertices) < 3:
        return None

    return _convex_hull_xy(top_vertices[:, :2])


def build_default_sequence():
    """Return a fixed 3-phase demo sequence in local coordinates."""
    phase_spiral = MotionSegment(
        name="planar_spiral",
        duration_s=2.0,
        local_curve=lambda tau: planar_spiral_xy(tau, r0=0.00, r1=0.12, cycles=2.5),
    )

    phase_shear = MotionSegment(
        name="oscillatory_shear",
        duration_s=1.5,
        local_curve=lambda tau: oscillatory_shear_local_profiled(
            tau,
            ShearProfile(
                depth_max=0.05, depth_ramp_end=0.7, shear_amp=0.012, shear_cycles=5.0
            ),
        ),
    )

    phase_sweep = MotionSegment(
        name="planar_sweep",
        duration_s=1.5,
        local_curve=lambda tau: planar_sweep_x(tau, length=0.10, cycles=2.0),
    )

    return MotionSequence([phase_spiral, phase_shear, phase_sweep])


def _duration_scale_from_body(
    body,
    reference_size=0.10,
    debug=False,
    use_visual_aabb=False,
    apply_shape_scale=False,
):
    """Compute a scaling factor from the body's AABB size."""
    mins, maxs = body_local_aabb(
        body,
        use_visual=use_visual_aabb,
        apply_shape_scale=apply_shape_scale,
    )
    extents = maxs - mins
    diag = float(np.linalg.norm(extents))
    ref = float(reference_size)
    if ref <= 0.0:
        raise ValueError("reference_size must be positive")
    scale = max(diag, 1e-6) / ref
    if debug:
        print(f"[motion_presets] aabb_diag={diag:.4f} ref={ref:.4f} scale={scale:.3f}")
    return scale


def build_container_sequence(
    bowl_body,
    reference_size=0.10,
    debug=False,
    use_visual_aabb=True,
    apply_shape_scale=True,
    pattern="spiral",
    mix_duration_s=None,
):
    """Build a spiral sequence sized to a bowl-like object."""
    mins, maxs = body_local_aabb(bowl_body, apply_shape_scale=apply_shape_scale)
    size_x = maxs[0] - mins[0]
    size_y = maxs[1] - mins[1]
    size_z = maxs[2] - mins[2]
    radius_xy = 0.5 * min(size_x, size_y)
    z_min, z_max = mins[2], maxs[2]

    center_y = 0.5 * (mins[1] + maxs[1])
    surface_margin = 0.005
    start_offset = np.array(
        [0.0, center_y, maxs[2] + APPROACH_Z_EXTRA_CLEARANCE_M - surface_margin],
        dtype=float,
    )
    duration_scale = _duration_scale_from_body(
        bowl_body,
        reference_size=reference_size,
        debug=debug,
        apply_shape_scale=apply_shape_scale,
    )
    spiral_r1 = 0.75 * radius_xy
    if debug:
        print(
            "[motion_presets] params "
            f"radius_xy={radius_xy:.4f} size_z={size_z:.4f} "
            f"spiral_r1={spiral_r1:.4f}"
        )

    # Keep points inside a bowl-shaped vertical cylinder.
    def _bowl_constraint(q_local):
        return clamp_to_cylinder_xy(
            q_local, radius=radius_xy, z_min=z_min, z_max=z_max, margin=0.005
        )

    # Offset the local curve and apply the bowl constraint.
    def _with_offset(curve):
        return make_constrained_curve(
            lambda tau: curve(tau) + start_offset, _bowl_constraint
        )

    phase_spiral_container = MotionSegment(
        name="planar_spiral_bowl",
        duration_s=2.0 * duration_scale,
        local_curve=_with_offset(
            lambda tau: planar_spiral_xy(tau, r0=0.00, r1=spiral_r1, cycles=2.0)
        ),
    )

    stir_amp = max(0.005, 0.55 * radius_xy)
    stir_base_duration = max(1.0, 2.0 * duration_scale)
    if mix_duration_s is not None and float(mix_duration_s) > 0.0:
        total_duration = float(mix_duration_s)
    else:
        total_duration = stir_base_duration
    stir_loops = max(1, int(np.ceil(total_duration / stir_base_duration)))

    phase_stir_container = MotionSegment(
        name="continuous_stir_bowl",
        duration_s=total_duration,
        local_curve=_with_offset(
            lambda tau: oscillatory_shear_xy_profiled(
                tau,
                ShearXYProfile(
                    shear_amp=stir_amp,
                    shear_cycles=stir_loops,
                ),
            )
        ),
    )

    pattern = str(pattern).lower()
    if pattern in ("spiral", "planar_spiral"):
        return MotionSequence([phase_spiral_container])
    if pattern in ("stir", "continuous", "continuous_stir", "loop"):
        return MotionSequence([phase_stir_container])

    raise ValueError(f"Unknown pattern '{pattern}'")


def build_surface_sequence(
    surface_body,
    reference_size=0.10,
    debug=False,
    use_visual_aabb=True,
    apply_shape_scale=True,
    technique="wipe",
    pattern=None,
):
    """Build a planar sequence on a surface or object (e.g., countertop, cutting)."""
    mins, maxs = body_local_aabb(
        surface_body,
        use_visual=use_visual_aabb,
        apply_shape_scale=apply_shape_scale,
    )
    size_x = maxs[0] - mins[0]
    size_y = maxs[1] - mins[1]
    size_z = maxs[2] - mins[2]
    radius_xy = 0.45 * min(size_x, size_y)

    center_x = 0.5 * (mins[0] + maxs[0])
    center_y = 0.5 * (mins[1] + maxs[1])
    surface_margin = 0.005
    start_offset = np.array(
        [
            center_x,
            center_y,
            maxs[2] + APPROACH_Z_EXTRA_CLEARANCE_M - surface_margin,
        ],
        dtype=float,
    )
    duration_scale = _duration_scale_from_body(
        surface_body,
        reference_size=reference_size,
        debug=debug,
        use_visual_aabb=use_visual_aabb,
        apply_shape_scale=apply_shape_scale,
    )
    spiral_r1 = 0.9 * radius_xy
    shear_amp = 0.35 * radius_xy
    depth_max = 0.8 * size_z
    raster_width = 0.9 * size_x
    raster_height = 0.9 * size_y
    lane_spacing = max(0.02, 0.25 * float(reference_size))
    raster_lanes = max(4, int(np.ceil(raster_height / max(lane_spacing, 1e-6))))
    if debug:
        print(
            "[motion_presets] surface params "
            f"radius_xy={radius_xy:.4f} spiral_r1={spiral_r1:.4f} "
            f"shear_amp={shear_amp:.4f} depth_max={depth_max:.4f} "
            f"raster_w={raster_width:.4f} raster_h={raster_height:.4f} "
            f"raster_lanes={raster_lanes}"
        )

    hull_xy = _top_surface_hull_xy(surface_body)
    if debug and hull_xy is not None:
        print(f"[motion_presets] top_surface_hull_vertices={len(hull_xy)}")

    def _with_hull_constraint(curve):
        if hull_xy is None:
            return lambda tau: curve(tau) + start_offset
        return make_constrained_curve(
            lambda tau: curve(tau) + start_offset,
            lambda q_local: _constrain_to_convex_hull_xy(q_local, hull_xy),
        )

    phase_spiral_surface = MotionSegment(
        name="planar_spiral_surface",
        duration_s=2.0 * duration_scale,
        local_curve=_with_hull_constraint(
            lambda tau: planar_spiral_xy(tau, r0=0.00, r1=spiral_r1, cycles=2.0)
        ),
    )

    phase_shear_surface = MotionSegment(
        name="oscillatory_shear_surface",
        duration_s=1.5 * duration_scale,
        local_curve=_with_hull_constraint(
            lambda tau: oscillatory_shear_xy_profiled(
                tau,
                ShearXYProfile(
                    shear_amp=shear_amp,
                    shear_cycles=5.0,
                ),
            )
        ),
    )

    phase_raster_surface = MotionSegment(
        name="planar_raster_surface",
        duration_s=2.0 * duration_scale,
        local_curve=_with_hull_constraint(
            lambda tau: planar_raster_xy(
                tau,
                width=raster_width,
                height=raster_height,
                lanes=raster_lanes,
            )
        ),
    )

    mode = pattern if pattern is not None else technique
    mode = str(mode).lower()

    if mode in ("wipe", "spiral", "planar_spiral"):
        return MotionSequence([phase_spiral_surface])
    if mode in ("shear", "oscillatory_shear", "shear_amp"):
        return MotionSequence([phase_shear_surface])
    if mode in ("spread", "raster", "planar_raster", "surface_cover"):
        return MotionSequence([phase_raster_surface])

    if pattern is not None:
        raise ValueError(f"Unknown pattern '{mode}'")
    raise ValueError(f"Unknown surface technique '{mode}'")


def build_cutting_sequence(
    food_body,
    reference_size=0.10,
    debug=False,
    use_visual_aabb=True,
    apply_shape_scale=True,
    technique="saw",
    slice_thickness=0.03,
    num_cuts_x=1,
):
    """Build a cutting sequence sized to a food object."""
    mins, maxs = body_local_aabb(
        food_body,
        use_visual=use_visual_aabb,
        apply_shape_scale=apply_shape_scale,
    )
    size_x = maxs[0] - mins[0]
    size_y = maxs[1] - mins[1]
    size_z = maxs[2] - mins[2]

    duration_scale = _duration_scale_from_body(
        food_body,
        reference_size=reference_size,
        debug=debug,
        apply_shape_scale=apply_shape_scale,
    )

    margin_x = min(0.01, 0.15 * size_x)
    margin_y = min(0.01, 0.10 * size_y)
    z_clearance = max(0.01, 0.25 * size_z) + APPROACH_Z_EXTRA_CLEARANCE_M
    z_top = maxs[2] + z_clearance
    z_cut = mins[2] + max(0.003, 0.05 * size_z)
    center_y = 0.5 * (mins[1] + maxs[1])

    usable_x = max(0.0, size_x - 2.0 * margin_x)
    requested_thickness = max(float(slice_thickness), 1e-4)
    x_anchor = mins[0] + margin_x + min(0.5 * requested_thickness, 0.5 * usable_x)
    x_max_anchor = maxs[0] - margin_x - min(0.5 * requested_thickness, 0.5 * usable_x)
    technique = str(technique).lower()

    n_cuts = max(1, int(num_cuts_x))
    if technique in ("halving", "halve"):
        x_anchors = [0.5 * (mins[0] + maxs[0])]
        z_cut = 0.5 * (mins[2] + maxs[2])
    elif n_cuts == 1:
        x_anchors = [x_anchor]
    else:
        if x_max_anchor <= x_anchor:
            x_anchors = [x_anchor] * n_cuts
        else:
            x_anchors = np.linspace(x_anchor, x_max_anchor, n_cuts).tolist()

    y_min = mins[1] + margin_y
    y_max = maxs[1] - margin_y
    saw_amp = 0.5 * max(y_max - y_min, 0.0)
    saw_cycles = max(2.0, round(size_y / max(reference_size, 1e-6)))

    if debug:
        print(
            "[motion_presets] cutting params "
            f"x_anchor={x_anchor:.4f} y_span={y_max - y_min:.4f} "
            f"z_top={z_top:.4f} z_cut={z_cut:.4f} technique={technique}"
        )

    shear_depth_max = maxs[2] - z_cut
    shear_amp = 0.5 * max(y_max - y_min, 0.0)

    technique = str(technique).lower()
    phases = []
    if technique in ("slice", "slicing", "halving", "halve"):
        for cut_idx, x_curr in enumerate(x_anchors):
            phases.extend(
                [
                    MotionSegment(
                        name=f"cut_approach_x{cut_idx}",
                        duration_s=0.8 * duration_scale,
                        local_curve=lambda tau, x_val=x_curr: np.array(
                            [
                                x_val,
                                center_y,
                                z_top + (maxs[2] - z_top) * float(tau),
                            ],
                            dtype=float,
                        ),
                    ),
                    MotionSegment(
                        name=f"cut_descend_x{cut_idx}",
                        duration_s=1.0 * duration_scale,
                        local_curve=lambda tau, x_val=x_curr: np.array(
                            [
                                x_val,
                                center_y,
                                maxs[2] + (z_cut - maxs[2]) * float(tau),
                            ],
                            dtype=float,
                        ),
                    ),
                    MotionSegment(
                        name=f"cut_retract_x{cut_idx}",
                        duration_s=0.8 * duration_scale,
                        local_curve=lambda tau, x_val=x_curr: np.array(
                            [
                                x_val,
                                center_y,
                                z_cut + (z_top - z_cut) * float(tau),
                            ],
                            dtype=float,
                        ),
                    ),
                ]
            )
    elif technique in ("saw", "sawing"):
        for cut_idx, x_curr in enumerate(x_anchors):
            phases.extend(
                [
                    MotionSegment(
                        name=f"cut_approach_x{cut_idx}",
                        duration_s=0.8 * duration_scale,
                        local_curve=lambda tau, x_val=x_curr: np.array(
                            [
                                x_val,
                                center_y,
                                z_top + (maxs[2] - z_top) * float(tau),
                            ],
                            dtype=float,
                        ),
                    ),
                    MotionSegment(
                        name=f"oscillatory_shear_x{cut_idx}",
                        duration_s=5.0 * duration_scale,
                        local_curve=lambda tau, x_val=x_curr: (
                            lambda q_local: np.array(
                                [
                                    x_val,
                                    center_y + q_local[0],
                                    maxs[2] + q_local[2],
                                ],
                                dtype=float,
                            )
                        )(
                            oscillatory_shear_local_profiled(
                                tau,
                                ShearProfile(
                                    depth_max=shear_depth_max,
                                    depth_ramp_end=0.7,
                                    shear_amp=shear_amp,
                                    shear_cycles=saw_cycles,
                                ),
                            )
                        ),
                    ),
                    MotionSegment(
                        name=f"cut_retract_x{cut_idx}",
                        duration_s=0.8 * duration_scale,
                        local_curve=lambda tau, x_val=x_curr: np.array(
                            [
                                x_val,
                                center_y,
                                z_cut + (z_top - z_cut) * float(tau),
                            ],
                            dtype=float,
                        ),
                    ),
                ]
            )
    else:
        raise ValueError(f"Unknown cutting technique '{technique}'")

    return MotionSequence(phases)


def build_pouring_sequence(
    source_body,
    *,
    target_body=None,
    reference_size=0.10,
    debug=False,
    use_visual_aabb=True,
    apply_shape_scale=True,
    pour_height=0.10,
    pour_offset_xy=(0.0, 0.0),
    approach_distance=0.08,
    retreat_distance=0.08,
    max_tilt=np.deg2rad(65.0),
    tilt_axis="y",
    approach_duration_s=1.2,
    tilt_in_duration_s=1.0,
    hold_duration_s=1.5,
    retreat_duration_s=1.2,
):
    """Build a simple pose-aware pouring sequence."""
    mins, maxs = body_local_aabb(
        source_body,
        use_visual=use_visual_aabb,
        apply_shape_scale=apply_shape_scale,
    )
    center_x = float(0.5 * (mins[0] + maxs[0]))
    center_y = float(0.5 * (mins[1] + maxs[1]))
    top_z = float(maxs[2])

    if target_body is not None:
        t_mins, t_maxs = body_local_aabb(
            target_body,
            use_visual=use_visual_aabb,
            apply_shape_scale=apply_shape_scale,
        )
        anchor_x = float(0.5 * (t_mins[0] + t_maxs[0])) + float(pour_offset_xy[0])
        anchor_y = float(0.5 * (t_mins[1] + t_maxs[1])) + float(pour_offset_xy[1])
        anchor_z = float(t_maxs[2]) + float(pour_height)
    else:
        anchor_x = center_x + float(pour_offset_xy[0])
        anchor_y = center_y + float(pour_offset_xy[1])
        anchor_z = top_z + float(pour_height)

    duration_scale = _duration_scale_from_body(
        source_body,
        reference_size=reference_size,
        debug=debug,
        use_visual_aabb=use_visual_aabb,
        apply_shape_scale=apply_shape_scale,
    )

    anchor_point = np.array([anchor_x, anchor_y, anchor_z], dtype=float)
    approach_point = np.array(
        [anchor_x - float(approach_distance), anchor_y, anchor_z], dtype=float
    )
    retreat_point = np.array(
        [anchor_x - float(retreat_distance), anchor_y, anchor_z], dtype=float
    )

    axis = str(tilt_axis).lower().strip()
    if axis != "y":
        raise ValueError(f"Unsupported pouring tilt axis '{tilt_axis}'")

    return MotionSequence(
        [
            MotionSegment(
                name="pour_approach",
                duration_s=float(approach_duration_s) * duration_scale,
                local_curve=lambda tau: approach_point
                + float(tau) * (anchor_point - approach_point),
                local_orientation_curve=fixed_rpy(),
            ),
            MotionSegment(
                name="pour_tilt_in",
                duration_s=float(tilt_in_duration_s) * duration_scale,
                local_curve=lambda tau: anchor_point,
                local_orientation_curve=lambda tau: rot_y(float(max_tilt) * float(tau)),
            ),
            MotionSegment(
                name="pour_hold",
                duration_s=float(hold_duration_s) * duration_scale,
                local_curve=lambda tau: anchor_point,
                local_orientation_curve=lambda tau: rot_y(float(max_tilt)),
            ),
            MotionSegment(
                name="pour_tilt_out_retreat",
                duration_s=float(retreat_duration_s) * duration_scale,
                local_curve=lambda tau: anchor_point
                + float(tau) * (retreat_point - anchor_point),
                local_orientation_curve=lambda tau: rot_y(
                    float(max_tilt) * (1.0 - float(tau))
                ),
            ),
        ]
    )
