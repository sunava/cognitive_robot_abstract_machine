from __future__ import annotations

from typing import Iterable

from geometry_msgs.msg import PoseStamped

from pycram.demos.thesis.actions.phase_runtime import CompilerRegistry, PrimitiveFamily
from pycram.demos.thesis.primitives.material_transfer import (
    BoundaryAnchor,
    DischargeSpec,
    ShakeSpec,
    compile_boundary_discharge,
    compile_boundary_shake,
)
from pycram.demos.thesis.primitives.seperation_devision import (
    SeparationSpec,
    compile_separation_contact,
)
from pycram.demos.thesis.primitives.surface_interaction import (
    ScrubSpec,
    SurfaceAnchor,
    SurfacePlane,
    SweepSpec,
    compile_scrub_circle,
    compile_wipe_raster_scrub,
)
from pycram.demos.thesis.primitives.volume_agitation import (
    AgitationSpec,
    VolumeAnchor,
    compile_volume_agitation,
)


def _compile_discharge(
    anchor: BoundaryAnchor, spec: DischargeSpec
) -> Iterable[PoseStamped]:
    return compile_boundary_discharge(anchor, spec)


def _compile_shake(anchor: BoundaryAnchor, spec: ShakeSpec) -> Iterable[PoseStamped]:
    return compile_boundary_shake(anchor, spec)


def _compile_sep(anchor: dict, spec: SeparationSpec) -> Iterable[PoseStamped]:
    return compile_separation_contact(
        frame_id=anchor["frame_id"],
        p0=anchor["p0"],
        n=anchor["n"],
        spec=spec,
        t1=anchor.get("t1"),
        t2=anchor.get("t2"),
    )


def _compile_scrub(anchor: SurfaceAnchor, spec_surface: dict) -> Iterable[PoseStamped]:
    surface: SurfacePlane = spec_surface["surface"]
    spec: ScrubSpec = spec_surface["scrub"]
    margin: float = float(spec_surface.get("margin", 0.0))
    return compile_scrub_circle(anchor, surface, spec, margin=margin)


def _compile_wipe(anchor: dict, spec_surface: dict) -> Iterable[PoseStamped]:
    surface: SurfacePlane = spec_surface["surface"]
    sweep: SweepSpec = spec_surface["sweep"]
    scrub: ScrubSpec = spec_surface["scrub"]
    return compile_wipe_raster_scrub(surface, sweep, scrub)


def _compile_agitation(anchor: VolumeAnchor, spec_vol: dict) -> Iterable[PoseStamped]:
    spec: AgitationSpec = spec_vol["agitation"]
    volume = spec_vol.get("volume")
    epsilon: float = float(spec_vol.get("epsilon", 0.0))
    return compile_volume_agitation(anchor, spec, volume=volume, epsilon=epsilon)


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
