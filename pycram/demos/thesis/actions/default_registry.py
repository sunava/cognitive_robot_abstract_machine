"""Default compiler registry for thesis action phases."""

from __future__ import annotations

from typing import Iterable

from geometry_msgs.msg import PoseStamped

from pycram.demos.thesis.actions.phase_runtime import CompilerRegistry, PrimitiveFamily
from pycram.demos.thesis.primitives.material_transfer import (
    DischargeSpec,
    ShakeSpec,
    compile_boundary_discharge,
    compile_boundary_shake,
)
from pycram.demos.thesis.primitives.seperation_devision import (
    compile_separation_contact,
)
from pycram.demos.thesis.primitives.surface_interaction import (
    compile_scrub_circle,
    compile_wipe_raster_scrub,
)
from pycram.demos.thesis.primitives.volume_agitation import (
    compile_volume_agitation,
)
from pycram.demos.thesis.actions.phase_parameters import (
    SeparationContactAnchor,
    SeparationContactParameters,
    SurfaceScrubParameters,
    SurfaceWipeParameters,
    VolumeAgitationParameters,
)


def _compile_discharge(anchor, spec: DischargeSpec) -> Iterable[PoseStamped]:
    """Compile boundary discharge poses."""
    return compile_boundary_discharge(anchor, spec)


def _compile_shake(anchor, spec: ShakeSpec) -> Iterable[PoseStamped]:
    """Compile boundary shake poses."""
    return compile_boundary_shake(anchor, spec)


def _compile_sep(
    anchor: SeparationContactAnchor, params: SeparationContactParameters
) -> Iterable[PoseStamped]:
    """Compile separation contact poses."""
    return compile_separation_contact(
        frame_id=anchor.frame_id,
        p0=anchor.p0,
        n=anchor.n,
        spec=params.spec,
        t1=anchor.t1,
        t2=anchor.t2,
    )


def _compile_scrub(anchor, params: SurfaceScrubParameters) -> Iterable[PoseStamped]:
    """Compile scrub circle poses."""
    return compile_scrub_circle(
        anchor,
        params.surface,
        params.scrub,
        margin=float(params.margin),
    )


def _compile_wipe(anchor, params: SurfaceWipeParameters) -> Iterable[PoseStamped]:
    """Compile raster wipe poses."""
    return compile_wipe_raster_scrub(params.surface, params.sweep, params.scrub)


def _compile_agitation(
    anchor, params: VolumeAgitationParameters
) -> Iterable[PoseStamped]:
    """Compile volume agitation poses."""
    return compile_volume_agitation(
        anchor,
        params.agitation,
        volume=params.volume,
        epsilon=float(params.epsilon),
    )


DEFAULT_REGISTRY = CompilerRegistry(
    compilers={
        PrimitiveFamily.MATERIAL_TRANSFER_DISCHARGE: _compile_discharge,
        PrimitiveFamily.MATERIAL_TRANSFER_SHAKE: _compile_shake,
        PrimitiveFamily.SEPARATION_CONTACT: _compile_sep,
        PrimitiveFamily.SURFACE_SCRUB_CIRCLE: _compile_scrub,
        PrimitiveFamily.SURFACE_WIPE_RASTER_SCRUB: _compile_wipe,
        PrimitiveFamily.VOLUME_AGITATION: _compile_agitation,
    }
)
