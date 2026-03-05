import numpy as np

from demos.thesis_new.thesis_math.motion_models import MotionSegment, MotionSequence
from demos.thesis_new.thesis_math.motion_profiles import (
    ShearProfile,
    ShearXYProfile,
    clamp_to_cylinder_xy,
    make_constrained_curve,
    oscillatory_shear_local_profiled,
    oscillatory_shear_xy_profiled,
    planar_raster_xy,
    planar_spiral_xy,
    planar_sweep_x,
)
from demos.thesis_new.thesis_math.world_utils import body_local_aabb


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
    body, reference_size=0.10, debug=False, apply_shape_scale=False
):
    """Compute a scaling factor from the body's AABB size."""
    mins, maxs = body_local_aabb(
        body, apply_shape_scale=apply_shape_scale
    )
    extents = maxs - mins
    diag = float(np.linalg.norm(extents))
    ref = float(reference_size)
    if ref <= 0.0:
        raise ValueError("reference_size must be positive")
    scale = max(diag, 1e-6) / ref
    if debug:
        print(
            f"[motion_presets] aabb_diag={diag:.4f} ref={ref:.4f} scale={scale:.3f}"
        )
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
    mins, maxs = body_local_aabb(
        bowl_body, apply_shape_scale=apply_shape_scale
    )
    size_x = maxs[0] - mins[0]
    size_y = maxs[1] - mins[1]
    size_z = maxs[2] - mins[2]
    radius_xy = 0.5 * min(size_x, size_y)
    z_min, z_max = mins[2], maxs[2]

    center_y = 0.5 * (mins[1] + maxs[1])
    surface_margin = 0.005
    start_offset = np.array([0.0, center_y, maxs[2] - surface_margin], dtype=float)
    duration_scale = _duration_scale_from_body(
        bowl_body,
        reference_size=reference_size,
        debug=debug,
        apply_shape_scale=apply_shape_scale,
    )
    spiral_r1 = 0.9 * radius_xy
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
            lambda tau: planar_spiral_xy(
                tau, r0=0.00, r1=spiral_r1, cycles=2.0
            )
        ),
    )

    stir_amp = max(0.005, 0.7 * radius_xy)
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
    pattern="spiral",
):
    """Build a planar sequence on a surface or object (e.g., countertop, cutting)."""
    mins, maxs = body_local_aabb(
        surface_body, apply_shape_scale=apply_shape_scale
    )
    size_x = maxs[0] - mins[0]
    size_y = maxs[1] - mins[1]
    size_z = maxs[2] - mins[2]
    radius_xy = 0.45 * min(size_x, size_y)

    center_x = 0.5 * (mins[0] + maxs[0])
    center_y = 0.5 * (mins[1] + maxs[1])
    surface_margin = 0.005
    start_offset = np.array(
        [center_x, center_y, maxs[2] - surface_margin], dtype=float
    )
    duration_scale = _duration_scale_from_body(
        surface_body,
        reference_size=reference_size,
        debug=debug,
        apply_shape_scale=apply_shape_scale,
    )
    spiral_r1 = 0.9 * radius_xy
    shear_amp = 0.35 * radius_xy
    depth_max = 0.8 * size_z
    raster_width = 0.9 * size_x
    raster_height = 0.9 * size_y
    raster_lanes = max(2, int(np.ceil(raster_height / max(reference_size, 1e-6))))
    if debug:
        print(
            "[motion_presets] surface params "
            f"radius_xy={radius_xy:.4f} spiral_r1={spiral_r1:.4f} "
            f"shear_amp={shear_amp:.4f} depth_max={depth_max:.4f} "
            f"raster_w={raster_width:.4f} raster_h={raster_height:.4f} "
            f"raster_lanes={raster_lanes}"
        )

    phase_spiral_surface = MotionSegment(
        name="planar_spiral_surface",
        duration_s=2.0 * duration_scale,
        local_curve=lambda tau: planar_spiral_xy(
            tau, r0=0.00, r1=spiral_r1, cycles=2.0
        )
        + start_offset,
    )

    phase_shear_surface = MotionSegment(
        name="oscillatory_shear_surface",
        duration_s=1.5 * duration_scale,
        local_curve=lambda tau: oscillatory_shear_xy_profiled(
            tau,
            ShearXYProfile(
                shear_amp=shear_amp,
                shear_cycles=5.0,
            ),
        )
        + start_offset,
    )

    phase_raster_surface = MotionSegment(
        name="planar_raster_surface",
        duration_s=2.0 * duration_scale,
        local_curve=lambda tau: planar_raster_xy(
            tau,
            width=raster_width,
            height=raster_height,
            lanes=raster_lanes,
        )
        + start_offset,
    )

    pattern = str(pattern).lower()
    if pattern in ("spiral", "planar_spiral"):
        return MotionSequence([phase_spiral_surface])
    if pattern in ("shear", "oscillatory_shear", "shear_amp"):
        return MotionSequence([phase_shear_surface])
    if pattern in ("raster", "planar_raster", "surface_cover"):
        return MotionSequence([phase_raster_surface])

    raise ValueError(f"Unknown pattern '{pattern}'")


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
    """Build a slicing sequence sized to a food object."""
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
    z_clearance = max(0.01, 0.25 * size_z)
    z_top = maxs[2] + z_clearance
    z_cut = mins[2] + max(0.003, 0.05 * size_z)
    center_y = 0.5 * (mins[1] + maxs[1])

    usable_x = max(0.0, size_x - 2.0 * margin_x)
    requested_thickness = max(float(slice_thickness), 1e-4)
    x_anchor = mins[0] + margin_x + min(0.5 * requested_thickness, 0.5 * usable_x)
    x_max_anchor = maxs[0] - margin_x - min(0.5 * requested_thickness, 0.5 * usable_x)
    n_cuts = max(1, int(num_cuts_x))
    if n_cuts == 1:
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
    if technique in "slice":
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
