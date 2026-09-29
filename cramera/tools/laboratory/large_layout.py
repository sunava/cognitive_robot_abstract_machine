"""Metric room, stations and instrument placements for the expanded medical lab."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import pi

from spec import DeviceSpec, Finish, PartSpec


# %% Stations and independent instrument instances
class MedicalStation(StrEnum):
    """Task-oriented areas around the preserved preparation bench."""

    PREPARATION = "preparation"
    SAMPLE_HANDLING = "sample_handling"
    THERMAL_PROCESSING = "thermal_processing"
    MEASUREMENT = "measurement"
    DOCUMENTATION = "documentation"


@dataclass(frozen=True)
class StationView:
    """A named work area and a native Z-up inspection camera."""

    identifier: MedicalStation
    """Stable UI and plan station key."""
    label: str
    """English station title."""
    position: tuple[float, float, float]
    """Camera position in the laboratory world."""
    target: tuple[float, float, float]
    """Camera look-at point in the laboratory world."""


@dataclass(frozen=True)
class InstrumentPlacement:
    """One portable model instance at an authored workbench pose."""

    identifier: str
    """Source resource directory and instance identifier."""
    station: MedicalStation
    """Associated task area."""
    position: tuple[float, float, float]
    """Root position in the world at the resting joint configuration."""
    attachment_parent: str = "laboratory_root"
    """World body or interaction frame carrying this instance."""
    attachment_position: tuple[float, float, float] | None = None
    """Local attachment offset when mounted on another instrument."""

    @property
    def prefix(self) -> str:
        """Namespace preserving all original body names within a scene instance."""
        return self.identifier + "__"

    @property
    def pose(self) -> list[float]:
        """Native world translation followed by an identity quaternion."""
        return [*self.position, 0, 0, 0, 1]


# %% Architectural geometry
@dataclass(frozen=True)
class LargeLaboratoryLayout:
    """A seven-by-six-metre room keeping calibrated preparation geometry fixed."""

    width: float = 7.0
    """Clear room width in metres."""
    depth: float = 6.0
    """Clear room depth in metres."""
    back_wall: float = 3.2
    """Interior rear-wall Y coordinate."""
    work_height: float = 0.9
    """Shared bench-top support plane."""
    rear_bench_center: float = 2.5
    """Y centre of the new peripheral workbench bank."""
    rear_bench_depth: float = 0.75
    """Depth of the new workbench surfaces."""
    original_bench_back: float = 0.375
    """Preserved preparation worktop's rear edge."""

    @property
    def rear_bench_front(self) -> float:
        """Front edge of the peripheral bank bordering the clear robot aisle."""
        return self.rear_bench_center - self.rear_bench_depth / 2

    def stations(self) -> tuple[StationView, ...]:
        """Return camera views that keep each group of instruments in context."""
        return (
            StationView(
                MedicalStation.PREPARATION,
                "01 · Sample preparation",
                (2.0, -2.8, 1.85),
                (0, 0, 0.92),
            ),
            StationView(
                MedicalStation.SAMPLE_HANDLING,
                "02 · Sample handling",
                (-1.90, 1.10, 1.75),
                (-2.48, 2.40, 1.06),
            ),
            StationView(
                MedicalStation.THERMAL_PROCESSING,
                "03 · Thermal processing",
                (-0.15, 0.80, 1.98),
                (-1.05, 2.42, 1.18),
            ),
            StationView(
                MedicalStation.MEASUREMENT,
                "04 · Analysis and measurement",
                (2.0, 0.72, 1.93),
                (1.04, 2.42, 1.12),
            ),
            StationView(
                MedicalStation.DOCUMENTATION,
                "05 · Identification and labels",
                (3.24, 1.17, 1.63),
                (2.60, 2.43, 1.05),
            ),
        )

    def placements(self) -> tuple[InstrumentPlacement, ...]:
        """Place all fifteen reusable resources with space for their opening motion."""
        handling, thermal, measure, documents = (
            MedicalStation.SAMPLE_HANDLING,
            MedicalStation.THERMAL_PROCESSING,
            MedicalStation.MEASUREMENT,
            MedicalStation.DOCUMENTATION,
        )
        return (
            InstrumentPlacement("tube_capper", handling, (-2.86, 2.43, 0.9)),
            InstrumentPlacement("vortex_mixer", handling, (-2.49, 2.39, 0.9)),
            InstrumentPlacement("tip_rack", handling, (-2.21, 2.36, 0.9)),
            InstrumentPlacement("electronic_pipette", handling, (-2.04, 2.40, 0.915)),
            InstrumentPlacement("thermomixer", thermal, (-1.61, 2.43, 0.9)),
            InstrumentPlacement("pcr_thermocycler", thermal, (-1.12, 2.43, 0.9)),
            InstrumentPlacement("incubator", thermal, (-0.49, 2.47, 0.9)),
            InstrumentPlacement("microplate_reader", measure, (0.22, 2.45, 0.9)),
            InstrumentPlacement("microplate_96_well", measure, (0.66, 2.35, 0.9)),
            InstrumentPlacement("digital_microscope", measure, (1.06, 2.45, 0.9)),
            InstrumentPlacement("ph_meter", measure, (1.60, 2.40, 0.9)),
            InstrumentPlacement(
                "ph_probe",
                measure,
                (1.774, 2.357, 1.005),
                "ph_meter__ph_meter_probe_mount",
                (0, 0, 0),
            ),
            InstrumentPlacement("barcode_dock", documents, (2.18, 2.40, 0.8995)),
            InstrumentPlacement(
                "barcode_scanner",
                documents,
                (2.18, 2.369, 0.9405),
                "barcode_dock__barcode_dock_scanner_mount",
                (0, -0.023, -0.003),
            ),
            InstrumentPlacement("label_printer", documents, (2.76, 2.47, 0.9)),
        )

    def room_definition(self) -> DeviceSpec:
        """Author a spacious shell and three independent supporting bench sections."""
        model = DeviceSpec(
            "medical_lab_room",
            "AICOR Medical Laboratory Room",
            "Expanded room and workstations surrounding the original calibrated preparation bench.",
        )
        shell = model.part("shell", mass=200)
        shell.box(
            "Floor",
            (0, 0.2, -0.04),
            (self.width, self.depth, 0.08),
            Finish.PALE_TEAL,
            bevel=0.002,
        )
        shell.box(
            "Back wall",
            (0, self.back_wall + 0.04, 1.45),
            (self.width, 0.08, 2.9),
            Finish.WHITE,
            bevel=0.003,
        )
        shell.box(
            "Left wall",
            (-self.width / 2 - 0.04, 0.2, 1.45),
            (0.08, self.depth, 2.9),
            Finish.WHITE,
            bevel=0.003,
        )
        shell.box(
            "Right lower wall",
            (self.width / 2 + 0.04, 0.2, 0.45),
            (0.08, self.depth, 0.9),
            Finish.WHITE,
            bevel=0.003,
        )
        shell.box(
            "Right upper wall",
            (self.width / 2 + 0.04, 0.2, 2.61),
            (0.08, self.depth, 0.58),
            Finish.WHITE,
            bevel=0.003,
        )
        shell.box(
            "Observation glazing",
            (self.width / 2 + 0.04, 0.2, 1.61),
            (0.014, self.depth, 1.42),
            Finish.GLASS,
            bevel=0.001,
        )
        for longitudinal in (-2.7, -1.3, 0.2, 1.7, 3.1):
            shell.box(
                f"Window mullion {longitudinal}",
                (3.54, longitudinal, 1.61),
                (0.055, 0.022, 1.44),
                Finish.STEEL,
                bevel=0.002,
            )
        shell.box(
            "Rear skirting",
            (0, 3.191, 0.075),
            (6.98, 0.016, 0.15),
            Finish.STEEL,
            bevel=0.002,
        )
        shell.box(
            "Left skirting",
            (-3.491, 0.2, 0.075),
            (0.016, 5.98, 0.15),
            Finish.STEEL,
            bevel=0.002,
        )
        for horizontal in (-3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0):
            shell.box(
                f"Rear panel seam {horizontal}",
                (horizontal, 3.197, 1.49),
                (0.002, 0.002, 2.68),
                Finish.PALE_TEAL,
                bevel=0,
                collision=False,
            )
        shell.logo((0, 3.191, 2.35), 0.54)
        shell.label(
            "MEDICAL LABORATORY", (0, 3.19, 2.15), 0.065, Finish.TEAL, width=1.45
        )
        shell.label(
            "SAMPLE HANDLING", (-2.39, 3.188, 1.84), 0.045, Finish.TEAL, width=1.40
        )
        shell.label(
            "THERMAL PROCESSING", (-0.90, 3.188, 1.84), 0.039, Finish.TEAL, width=1.30
        )
        shell.label("ANALYSIS", (1.02, 3.188, 1.84), 0.050, Finish.TEAL, width=1.0)
        shell.label(
            "IDENTIFICATION", (2.62, 3.188, 1.84), 0.040, Finish.TEAL, width=1.1
        )
        for index, horizontal in enumerate((-2.1, 0, 2.1), start=1):
            bench = model.part(f"bench_{index}", parent="shell", mass=70)
            self._bench(bench, horizontal, index)
            model.frame(
                f"bench_{index}_surface",
                f"bench_{index}",
                (horizontal, self.rear_bench_center, self.work_height),
                "work_surface",
            )
        self._pipette_holder(model)
        for index, horizontal in enumerate((-2.1, 0, 2.1)):
            shell.box(
                f"Ceiling luminaire {index}",
                (horizontal, 1.65, 2.70),
                (1.55, 0.24, 0.045),
                Finish.WHITE,
                bevel=0.009,
            )
            shell.box(
                f"Light diffuser {index}",
                (horizontal, 1.65, 2.675),
                (1.48, 0.19, 0.005),
                Finish.PAPER,
                bevel=0.003,
                collision=False,
            )
            for side in (-0.57, 0.57):
                shell.cylinder(
                    f"Luminaire suspension {index} {side}",
                    (horizontal + side, 1.65, 2.81),
                    0.003,
                    0.22,
                    Finish.STEEL,
                    collision=False,
                )
        return model

    def _bench(self, part: PartSpec, horizontal: float, index: int) -> None:
        """Build a continuous hygienic worktop with recessed cabinet fronts."""
        part.box(
            f"Worktop {index}",
            (horizontal, self.rear_bench_center, 0.876),
            (2.1, self.rear_bench_depth, 0.048),
            Finish.WHITE,
            bevel=0.008,
        )
        part.box(
            f"Splashback {index}",
            (horizontal, 2.851, 0.956),
            (2.09, 0.025, 0.112),
            Finish.WHITE,
            bevel=0.003,
        )
        part.box(
            f"Cabinet case {index}",
            (horizontal, 2.52, 0.482),
            (2.04, 0.615, 0.716),
            Finish.WHITE,
            bevel=0.006,
        )
        part.box(
            f"Recessed cabinet plinth {index}",
            (horizontal, 2.54, 0.079),
            (1.96, 0.51, 0.12),
            Finish.TEAL,
            bevel=0.004,
        )
        for door, offset in enumerate((-0.765, -0.255, 0.255, 0.765)):
            part.box(
                f"Cabinet door {index} {door}",
                (horizontal + offset, 2.205, 0.485),
                (0.494, 0.018, 0.682),
                Finish.WHITE,
                bevel=0.006,
            )
            part.box(
                f"Cabinet pull {index} {door}",
                (horizontal + offset, 2.184, 0.720),
                (0.13, 0.022, 0.018),
                Finish.TEAL,
                bevel=0.005,
            )
        part.label(
            f"AICOR  /  WORKSTATION {index + 1:02d}",
            (horizontal, 2.123, 0.877),
            0.020,
            Finish.TEAL,
            width=0.68,
        )

    def _pipette_holder(self, model: DeviceSpec) -> None:
        """Support the handheld pipette underneath its existing finger rest."""
        holder = model.part(
            "pipette_holder", parent="shell", origin=(-2.04, 2.40, 0.9), mass=0.25
        )
        holder.box(
            "Pipette stand foot",
            (0.035, 0.025, 0.009),
            (0.12, 0.12, 0.018),
            Finish.PALE_TEAL,
            bevel=0.005,
        )
        holder.box(
            "Pipette stand upright",
            (0.065, 0.04, 0.140),
            (0.014, 0.025, 0.270),
            Finish.STEEL,
            bevel=0.003,
        )
        holder.box(
            "Pipette support",
            (0.026, 0.028, 0.2705),
            (0.091, 0.018, 0.016),
            Finish.TEAL,
            bevel=0.003,
        )
