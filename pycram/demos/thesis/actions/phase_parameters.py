"""Parameter and anchor dataclasses for phase compilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pycram.demos.thesis.primitives.seperation_devision import SeparationSpec
from pycram.demos.thesis.primitives.surface_interaction import (
    ScrubSpec,
    SurfacePlane,
    SweepSpec,
)
from pycram.demos.thesis.primitives.volume_agitation import AgitationSpec
from pycram.demos.thesis.geometry.volume_models import ContainerVolumeModel
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(frozen=True)
class SeparationContactAnchor:
    """Anchor for separation contact primitives."""

    frame_id: Body
    p0: np.ndarray
    n: np.ndarray
    t1: Optional[np.ndarray] = None
    t2: Optional[np.ndarray] = None


@dataclass(frozen=True)
class SeparationContactParameters:
    """Parameters for separation contact primitives."""

    spec: SeparationSpec


@dataclass(frozen=True)
class SurfaceScrubParameters:
    """Parameters for surface scrub primitives."""

    surface: SurfacePlane
    scrub: ScrubSpec
    margin: float = 0.0


@dataclass(frozen=True)
class SurfaceWipeParameters:
    """Parameters for surface wipe primitives."""

    surface: SurfacePlane
    sweep: SweepSpec
    scrub: ScrubSpec


@dataclass(frozen=True)
class VolumeAgitationParameters:
    """Parameters for volume agitation primitives."""

    agitation: AgitationSpec
    volume: Optional[ContainerVolumeModel] = None
    epsilon: float = 0.0
