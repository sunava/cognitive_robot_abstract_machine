from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np


class PrimitiveFamily(str, Enum):
    MATERIAL_TRANSFER = "material_transfer"
    SEPARATION_AND_DIVISION = "separation_and_division"
    SURFACE_INTERACTION = "surface_interaction"
    AGGREGATION_AND_MIXING = "aggregation_and_mixing"
    FREE_MOTION = "free_motion"


@dataclass(frozen=True)
class ArticulationModel:
    kind: str
    reference_frame: Any
    properties: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContainerArticulationModel(ArticulationModel):
    radius_xy: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0
    start_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))


@dataclass(frozen=True)
class SurfaceArticulationModel(ArticulationModel):
    center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    normal: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float)
    )
    extents_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))


@dataclass(frozen=True)
class CuttingArticulationModel(ArticulationModel):
    surface_point: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    normal: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -1.0], dtype=float)
    )
    tangent_axis: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=float)
    )
    cut_depth: float = 0.0


@dataclass(frozen=True)
class MaterialTransferArticulationModel(ArticulationModel):
    source_opening: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    target_opening: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    flow_direction: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float)
    )
    discharge_axis: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -1.0], dtype=float)
    )


@dataclass(frozen=True)
class OAAMPhase:
    family: PrimitiveFamily
    anchor: Any
    parameters: Mapping[str, Any] = field(default_factory=dict)
    invariant: str = ""
    articulation_model: ArticulationModel | None = None
