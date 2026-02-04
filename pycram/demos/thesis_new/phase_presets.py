import numpy as np

from semantic_digital_twin.semantic_annotations.position_descriptions import (
    SemanticPositionDescription,
    HorizontalSemanticDirection,
    VerticalSemanticDirection,
)

from demos.thesis_new.phase_models import Phase, PhaseSequence
from demos.thesis_new.phase_profiles import (
    ShearProfile,
    oscillatory_shear_local_profiled,
    planar_spiral_xy,
    planar_sweep_x,
    clamp_to_cylinder_xy,
    make_constrained_curve,
)
from demos.thesis_new.world_utils import body_local_aabb, sample_semantic_yz


def build_default_sequence():
    phase_spiral = Phase(
        name="planar_spiral",
        duration_s=2.0,
        local_curve=lambda tau: planar_spiral_xy(tau, r0=0.00, r1=0.12, cycles=2.5),
    )

    phase_shear = Phase(
        name="oscillatory_shear",
        duration_s=1.5,
        local_curve=lambda tau: oscillatory_shear_local_profiled(
            tau,
            ShearProfile(
                depth_max=0.05, depth_ramp_end=0.7, shear_amp=0.012, shear_cycles=5.0
            ),
        ),
    )

    phase_sweep = Phase(
        name="planar_sweep",
        duration_s=1.5,
        local_curve=lambda tau: planar_sweep_x(tau, length=0.10, cycles=2.0),
    )

    return PhaseSequence([phase_spiral, phase_shear, phase_sweep])


def build_bowl_sequence(bowl_body):
    mins, maxs = body_local_aabb(bowl_body)
    radius_xy = 0.5 * min(maxs[0] - mins[0], maxs[1] - mins[1])
    z_min, z_max = mins[2], maxs[2]

    sem_start = SemanticPositionDescription(
        horizontal_direction_chain=[HorizontalSemanticDirection.LEFT],
        vertical_direction_chain=[VerticalSemanticDirection.CENTER],
    )
    y0, z0 = sample_semantic_yz(bowl_body, sem_start)
    start_offset = np.array([0.0, y0, z0], dtype=float)

    def _bowl_constraint(q_local):
        return clamp_to_cylinder_xy(
            q_local, radius=radius_xy, z_min=z_min, z_max=z_max, margin=0.005
        )

    def _with_offset(curve):
        return make_constrained_curve(
            lambda tau: curve(tau) + start_offset, _bowl_constraint
        )

    phase_spiral_bowl = Phase(
        name="planar_spiral_bowl",
        duration_s=2.0,
        local_curve=_with_offset(
            lambda tau: planar_spiral_xy(tau, r0=0.00, r1=0.8 * radius_xy, cycles=2.0)
        ),
    )
    phase_shear_bowl = Phase(
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
    phase_sweep_bowl = Phase(
        name="planar_sweep_bowl",
        duration_s=1.5,
        local_curve=_with_offset(
            lambda tau: planar_sweep_x(tau, length=0.10, cycles=2.0)
        ),
    )

    return PhaseSequence([phase_spiral_bowl, phase_shear_bowl, phase_sweep_bowl])
