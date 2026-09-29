"""
Inserted plates and articulated chamber parts remain mechanically accessible.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


# %% Standalone model loading
@pytest.fixture
def chamber_models(monkeypatch: pytest.MonkeyPatch):
    """
    Load original model specifications without requiring Blender.
    """
    directory = (
        Path(__file__).resolve().parents[2] / "cramera/tools/laboratory/instruments"
    )
    monkeypatch.syspath_prepend(str(directory))
    models = importlib.import_module("chamber_devices").build_devices()
    return {model.identifier: model for model in models}


# %% Closure clearances
class TestPlateInsertion:
    """
    Plate placement frames describe physically unobstructed spaces.
    """

    def test_closed_heated_lid_clears_seated_plate(self, chamber_models) -> None:
        """
        The heated pressure pad remains above an inserted plate rim.
        """
        thermal_cycler = chamber_models["pcr_thermocycler"]
        plate = chamber_models["microplate_96_well"]
        block = next(
            part for part in thermal_cycler.parts if part.name == "reaction_block"
        )
        lid = next(part for part in thermal_cycler.parts if part.name == "heated_lid")
        seat = next(
            frame for frame in thermal_cycler.frames if frame.name == "plate_seat"
        )
        pad = next(shape for shape in lid.shapes if shape.name == "heated_pressure_pad")
        plate_rim = next(
            shape for shape in plate.parts[0].shapes if shape.name == "rim_front"
        )
        seated_top = (
            block.origin[2]
            + seat.position[2]
            + plate_rim.center[2]
            + plate_rim.dimensions[2] / 2
        )
        pad_bottom = lid.origin[2] + pad.center[2] - pad.dimensions[2] / 2
        assert seated_top <= pad_bottom

    def test_plate_fits_reader_well(self, chamber_models) -> None:
        """
        A plate fits between all four drawer retaining walls.
        """
        reader = chamber_models["microplate_reader"]
        plate = chamber_models["microplate_96_well"]
        drawer = next(part for part in reader.parts if part.name == "plate_drawer")
        plate_floor = next(
            shape for shape in plate.parts[0].shapes if shape.name == "bottom"
        )
        right = next(
            shape for shape in drawer.shapes if shape.name == "plate_side_right"
        )
        back = next(shape for shape in drawer.shapes if shape.name == "plate_end_back")
        assert plate_floor.dimensions[0] / 2 < right.center[0] - right.dimensions[0] / 2
        assert plate_floor.dimensions[1] / 2 < back.center[1] - back.dimensions[1] / 2

    def test_microplate_wells_reach_the_floor(self, chamber_models) -> None:
        """
        Every hollow well wall closes against the plate floor.
        """
        plate = chamber_models["microplate_96_well"]
        floor = next(shape for shape in plate.parts[0].shapes if shape.name == "bottom")
        top = floor.center[2] + floor.dimensions[2] / 2
        wells = [
            shape for shape in plate.parts[0].shapes if shape.name.startswith("well_")
        ]
        assert len(wells) == 96
        for well in wells:
            assert well.center[2] - well.dimensions[2] / 2 <= top
            assert well.dimensions[1] > 0


# %% Independent descriptions
class TestIndependentDescriptions:
    """
    Every authored instrument retains a complete local articulation hierarchy.
    """

    def test_all_device_descriptions_validate(self, chamber_models) -> None:
        """
        All frame parents, body masses and joint ranges satisfy the export contract.
        """
        for model in chamber_models.values():
            model.validate()


# %% Visible mechanical supports
class TestClampSupports:
    """
    Moving jaws stay physically connected to the stationary carriage housing.
    """

    def test_each_jaw_reaches_its_slide_housing(self, chamber_models) -> None:
        """
        The moving assembly extends continuously from its jaw into the rear support.
        """
        capper = chamber_models["tube_capper"]
        support = next(
            shape for shape in capper.parts[0].shapes if shape.name == "clamp_support"
        )
        support_front = support.center[1] - support.dimensions[1] / 2
        for part in capper.parts:
            if not part.name.startswith("clamp_"):
                continue
            boxes = [
                shape
                for shape in part.shapes
                if shape.collision and shape.kind == "box"
            ]
            jaw = next(shape for shape in boxes if shape.name == "jaw_body")
            reached = {jaw.name}
            pending = [jaw]
            while pending:
                connected = pending.pop()
                for candidate in boxes:
                    if candidate.name in reached:
                        continue
                    touching = all(
                        abs(connected.center[axis] - candidate.center[axis])
                        <= (connected.dimensions[axis] + candidate.dimensions[axis]) / 2
                        for axis in range(3)
                    )
                    if touching:
                        reached.add(candidate.name)
                        pending.append(candidate)
            rear_edge = max(
                part.origin[1] + shape.center[1] + shape.dimensions[1] / 2
                for shape in boxes
                if shape.name in reached
            )
            assert rear_edge >= support_front


class TestWellIndexMarkings:
    """
    Address labels remain visibly printed above their supporting plate rim.
    """

    def test_row_letters_are_above_the_rim(self, chamber_models) -> None:
        """
        Row names cannot be hidden inside the left perimeter wall.
        """
        plate = chamber_models["microplate_96_well"].parts[0]
        rim = next(shape for shape in plate.shapes if shape.name == "rim_left")
        rim_top = rim.center[2] + rim.dimensions[2] / 2
        letters = [
            shape
            for shape in plate.shapes
            if len(shape.text) == 1 and shape.text.isalpha()
        ]
        assert len(letters) == 8
        for letter in letters:
            assert letter.center[2] > rim_top
