"""
A fourth independent glass supplies the centrifuge's second opposing pair.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import pytest

from cramera.laboratory_liquid import LiquidField, TubeKey
from cramera.laboratory_physics import LaboratoryPhysics, PhysicsField
from cramera.laboratory_world import (
    LaboratoryAsset,
    LaboratoryBody,
    LaboratoryEnvironment,
    LaboratoryField,
    LaboratorySlot,
)
from semantic_digital_twin.world_description.connections import Connection6DoF

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics import physics
from .test_laboratory_world import laboratory_bundle


# %% authored reserve glass
def test_reserve_uses_an_independent_body_and_shared_clear_glass_mesh(
    laboratory_directory: Path,
) -> None:
    """
    The rear-right rack slot supplies a fourth glass without duplicating its mesh.
    """
    scene = json.loads((laboratory_directory / LaboratoryAsset.SCENE).read_text())
    objects = {
        item[LaboratoryField.KEY]: item for item in scene[LaboratoryField.OBJECTS]
    }
    reserve = objects[LaboratoryBody.RESERVE_TUBE]
    description = LaboratoryEnvironment(
        bundle_directory=laboratory_directory
    ).description
    assert description.occupancy[LaboratorySlot.B3] == LaboratoryBody.RESERVE_TUBE
    assert len(description.occupancy) == 4
    assert LaboratorySlot.A3 not in description.occupancy
    assert LaboratorySlot.B2 not in description.occupancy
    np.testing.assert_allclose(
        reserve[LaboratoryField.SPAWN][:3],
        description.slots[LaboratorySlot.B3].to_np()[:3, 3],
    )
    reserve_urdf = ElementTree.parse(
        laboratory_directory / reserve[LaboratoryField.URDF]
    )
    clear_urdf = ElementTree.parse(
        laboratory_directory / objects[LaboratoryBody.CLEAR_TUBE][LaboratoryField.URDF]
    )
    assert reserve_urdf.find("link").attrib["name"] == LaboratoryBody.RESERVE_TUBE
    assert (
        reserve_urdf.find(".//visual/geometry/mesh").attrib
        == clear_urdf.find(".//visual/geometry/mesh").attrib
    )


def test_reserve_is_a_free_body_in_cram_world(laboratory_bundle: Path) -> None:
    """
    General CRAM world loading exposes the spare vessel as a manipulable body.
    """
    laboratory = LaboratoryEnvironment(bundle_directory=laboratory_bundle)
    world = laboratory.create_world()
    reserve = world.get_body_by_name(LaboratoryBody.RESERVE_TUBE)
    assert isinstance(reserve.parent_connection, Connection6DoF)
    np.testing.assert_allclose(
        reserve.global_pose.to_np(),
        laboratory.description.slots[LaboratorySlot.B3].to_np(),
    )


def test_packaged_world_includes_the_reserve_glass() -> None:
    """
    The installed portable world also supplies all four centrifuge samples.
    """
    packaged = (
        Path(__file__).resolve().parents[2]
        / "cramera/src/cramera/resources/aicor_medical_lab"
    )
    world = LaboratoryEnvironment(bundle_directory=packaged).create_world()
    assert isinstance(
        world.get_body_by_name(LaboratoryBody.RESERVE_TUBE).parent_connection,
        Connection6DoF,
    )


# %% physical contents and support
def test_reserve_can_be_filled_and_reset_without_changing_original_contents(
    physics: LaboratoryPhysics,
) -> None:
    """
    An initially empty reserve has liquid mass and leaves donor fills unchanged.
    """
    key = LaboratoryBody.RESERVE_TUBE
    initial = physics.snapshot()[PhysicsField.LIQUID]
    assert initial[LiquidField.TUBES][key][LiquidField.VOLUME] == 0
    donor_total = sum(
        initial[LiquidField.TUBES][donor][LiquidField.VOLUME]
        for donor in (TubeKey.AMBER, TubeKey.TEAL)
    )
    assert initial[LiquidField.TOTAL] == pytest.approx(donor_total)
    identifier = physics.objects[key].body_id
    empty_mass = physics.model.body_mass[identifier]
    volume = 5.0
    physics.fill_liquid(key, volume)
    physics.step(1000)
    filled = physics.snapshot()[PhysicsField.LIQUID]
    assert filled[LiquidField.TUBES][key][LiquidField.VOLUME] == pytest.approx(volume)
    assert physics.model.body_mass[identifier] == pytest.approx(
        empty_mass + volume / 1000
    )
    for donor in (TubeKey.CLEAR, TubeKey.AMBER, TubeKey.TEAL):
        assert (
            filled[LiquidField.TUBES][donor][LiquidField.VOLUME]
            == initial[LiquidField.TUBES][donor][LiquidField.VOLUME]
        )
    physics.reset()
    assert physics.snapshot()[PhysicsField.LIQUID] == initial


def test_four_glasses_remain_supported_in_their_authored_rack_slots(
    physics: LaboratoryPhysics,
) -> None:
    """
    The added reserve neither falls through the rack nor displaces its neighbours.
    """
    initial = physics.snapshot()[PhysicsField.OBJECTS]
    assert LaboratoryBody.RESERVE_TUBE in initial
    physics.step(2000)
    final = physics.snapshot()[PhysicsField.OBJECTS]
    for key in initial:
        np.testing.assert_allclose(final[key][:3], initial[key][:3], atol=0.001)
