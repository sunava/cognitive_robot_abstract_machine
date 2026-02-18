import numpy as np

from pycram.testing import setup_world
from demos.thesis_new.thesis_math.frame_provider import WorldTransformFrameProvider
from demos.thesis_new.old.Phasenbausteineinwelt import (
    try_get_body,
    body_local_aabb,
    sample_semantic_yz,
    clamp_to_cylinder_xy,
    make_constrained_curve,
    planar_spiral_xy,
    oscillatory_shear_local_profiled,
    planar_sweep_x,
    ShearProfile,
    MotionSegment,
    MotionSequence,
    make_identity_pose_stamped,
    SemanticPositionDescription,
    HorizontalSemanticDirection,
    VerticalSemanticDirection,
)


def main():
    """Sample a bowl-constrained sequence and validate AABB containment."""
    world = setup_world()

    bowl_name = "bowl.stl"
    bowl_body = try_get_body(world, bowl_name)
    if bowl_body is None:
        print(
            f"[info] body '{bowl_name}' not found, skipping object-dependent example."
        )
        return

    mins, maxs = body_local_aabb(bowl_body)
    radius_xy = 0.5 * min(maxs[0] - mins[0], maxs[1] - mins[1])
    z_min, z_max = mins[2], maxs[2]

    sem_start = SemanticPositionDescription(
        horizontal_direction_chain=[HorizontalSemanticDirection.LEFT],
        vertical_direction_chain=[VerticalSemanticDirection.CENTER],
    )
    y0, z0 = sample_semantic_yz(bowl_body, sem_start)
    start_offset = np.array([0.0, y0, z0], dtype=float)

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
        duration_s=2.0,
        local_curve=_with_offset(
            lambda tau: planar_spiral_xy(tau, r0=0.00, r1=0.8 * radius_xy, cycles=2.0)
        ),
    )
    phase_shear_container = MotionSegment(
        name="oscillatory_shear_bowl",
        duration_s=1.5,
        local_curve=_with_offset(
            lambda tau: oscillatory_shear_local_profiled(
                tau,
                ShearProfile(
                    depth_max=0.05,
                    depth_ramp_end=0.7,
                    shear_amp=0.012,
                    shear_cycles=5.0,
                ),
            )
        ),
    )
    phase_sweep_container = MotionSegment(
        name="planar_sweep_bowl",
        duration_s=1.5,
        local_curve=_with_offset(
            lambda tau: planar_sweep_x(tau, length=0.10, cycles=2.0)
        ),
    )

    seq_container = MotionSequence([phase_spiral_container, phase_shear_container, phase_sweep_container])
    prov_container = WorldTransformFrameProvider(
        world=world,
        source_frame=bowl_body,
        root_frame=world.root,
        make_identity_spatial=make_identity_pose_stamped,
    )
    t_bowl, P_container, id_container = seq_container.sample(prov_container, dt=0.01)

    # Basic validation: all points are inside the AABB in bowl local coords
    mins, maxs = body_local_aabb(bowl_body)
    local_ok = np.all((P_container >= mins - 1e-6) & (P_container <= maxs + 1e-6))
    print("Sampled points count:", len(P_container))
    print("Inside local AABB:", local_ok)


if __name__ == "__main__":
    main()
