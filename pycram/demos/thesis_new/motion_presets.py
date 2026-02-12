import numpy as np

from demos.thesis_new.motion_models import MotionSegment, MotionSequence
from demos.thesis_new.motion_profiles import (
    ShearProfile,
    oscillatory_shear_local_profiled,
    planar_spiral_xy,
    planar_sweep_x,
    clamp_to_cylinder_xy,
    make_constrained_curve,
)
from demos.thesis_new.world_utils import body_local_aabb


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
    body, reference_size=0.10, debug=False, use_visual=False, apply_shape_scale=False
):
    """Compute a scaling factor from the body's AABB size."""
    mins, maxs = body_local_aabb(
        body, use_visual=use_visual, apply_shape_scale=apply_shape_scale
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
):
    """Build a 3-phase sequence sized to a bowl-like object."""
    mins, maxs = body_local_aabb(
        bowl_body, use_visual=use_visual_aabb, apply_shape_scale=apply_shape_scale
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
        use_visual=use_visual_aabb,
        apply_shape_scale=apply_shape_scale,
    )
    spiral_r1 = 0.9 * radius_xy
    sweep_length = 0.9 * radius_xy
    shear_amp = 0.35 * radius_xy
    depth_max = 0.8 * size_z
    if debug:
        print(
            "[motion_presets] params "
            f"radius_xy={radius_xy:.4f} size_z={size_z:.4f} "
            f"spiral_r1={spiral_r1:.4f} sweep_len={sweep_length:.4f} "
            f"shear_amp={shear_amp:.4f} depth_max={depth_max:.4f}"
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
    phase_shear_container = MotionSegment(
        name="oscillatory_shear_bowl",
        duration_s=1.5 * duration_scale,
        local_curve=_with_offset(
            lambda tau: oscillatory_shear_local_profiled(
                tau,
                ShearProfile(
                    depth_max=depth_max,
                    depth_ramp_end=0.7,
                    shear_amp=shear_amp,
                    shear_cycles=5.0,
                ),
            )
        ),
    )
    phase_sweep_container = MotionSegment(
        name="planar_sweep_bowl",
        duration_s=1.5 * duration_scale,
        local_curve=_with_offset(
            lambda tau: planar_sweep_x(
                tau, length=sweep_length, cycles=2.0
            )
        ),
    )

    return MotionSequence([phase_spiral_container, phase_shear_container, phase_sweep_container])
