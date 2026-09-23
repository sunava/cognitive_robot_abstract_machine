"""
The portable CRAM laboratory includes independently articulated instruments.
"""

from __future__ import annotations

import json
import math
from enum import StrEnum

import numpy as np

from cramera.laboratory_world import LaboratoryAsset, LaboratoryEnvironment
from semantic_digital_twin.world_description.connections import RevoluteConnection


# %% portable articulation
class InstrumentField(StrEnum):
    """
    Authored metadata locating the instrument's independent bodies.
    """

    LABORATORY = "laboratory"
    """
    Manual source scene metadata.
    """

    CENTRIFUGE = "centrifuge"
    """
    Instrument metadata within the laboratory.
    """

    LID_BODY = "lidBody"
    """
    Name of the hinged protective cover.
    """

    ROTOR_BODY = "rotorBody"
    """
    Name of the continuously rotating sample carrier.
    """

    OPEN_ANGLE = "openAngle"
    """
    Maximum authored lid opening angle in radians.
    """


def test_packaged_centrifuge_has_independent_lid_and_rotor() -> None:
    """
    A clean CRAM world can move both instrument joints without a local cache.
    """
    laboratory = LaboratoryEnvironment()
    source = json.loads(
        (laboratory.bundle_directory / LaboratoryAsset.SCENE).read_text()
    )
    instrument = source[InstrumentField.LABORATORY][InstrumentField.CENTRIFUGE]
    world = laboratory.create_world()
    lid = world.get_body_by_name(instrument[InstrumentField.LID_BODY])
    rotor = world.get_body_by_name(instrument[InstrumentField.ROTOR_BODY])
    assert isinstance(lid.parent_connection, RevoluteConnection)
    assert isinstance(rotor.parent_connection, RevoluteConnection)
    closed = lid.global_pose.to_np().copy()
    initial_rotor = rotor.global_pose.to_np().copy()
    lid.parent_connection.position = instrument[InstrumentField.OPEN_ANGLE]
    opened = lid.global_pose.to_np().copy()
    assert not np.allclose(opened[:3, :3], closed[:3, :3])
    np.testing.assert_allclose(rotor.global_pose.to_np(), initial_rotor)
    rotor.parent_connection.position = math.pi / 2
    rotated = rotor.global_pose.to_np()
    np.testing.assert_allclose(rotated[:3, 0], initial_rotor[:3, 1], atol=1e-12)
    np.testing.assert_allclose(rotated[:3, 3], initial_rotor[:3, 3], atol=1e-12)
    np.testing.assert_allclose(lid.global_pose.to_np(), opened)
